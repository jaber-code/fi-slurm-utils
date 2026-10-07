use chrono::{DateTime, Duration, Local, Utc};
use fi_slurm::list::{SlurmIterator, vec_to_slurm_list};
use fi_slurm::site;
use fi_slurm::steps::StepId;
use fi_slurm::utils::c_str_to_string;
use fi_slurm_sys::{
    SLURMDB_JOB_FLAG_NOTSET, slurm_list_destroy, slurmdb_job_cond_t, slurmdb_job_rec_t,
    slurmdb_jobs_get, slurmdb_step_rec_t, xlist,
};
use thiserror::Error;

use crate::db::{DbConn, slurmdb_connect};

/// What can go wrong querying the accounting database for jobs
#[derive(Error, Debug)]
pub enum SacctError {
    #[error(
        "Database connection failed. Please ensure that SlurmDB is present and slurm_init has been run"
    )]
    DbConnError,
    #[error("SlurmDB returned no job list")]
    JobListNull,
}

/// The conditions of a job query, owning the Slurm lists handed over in it
struct JobCond {
    cond: slurmdb_job_cond_t,
}

impl JobCond {
    /// Selects every user's jobs that were eligible to run at some point between
    /// `usage_start` and `usage_end`, as `sacct --allusers -S start -E end` does
    fn new(usage_start: DateTime<Utc>, usage_end: DateTime<Utc>) -> Self {
        let mut cond: slurmdb_job_cond_t = unsafe { std::mem::zeroed() };

        // no cluster leaves slurmdbd to answer for the cluster this connection belongs to
        cond.cluster_list = unsafe { vec_to_slurm_list(site::cluster().clone().map(|c| vec![c])) };
        // zero would filter on the scheduling flags, matching no job at all; sacct sets this
        // unless asked for jobs by how they were scheduled
        cond.db_flags = SLURMDB_JOB_FLAG_NOTSET;
        cond.usage_start = usage_start.timestamp();
        cond.usage_end = usage_end.timestamp();
        // userid_list stays null for every user, and flags stays zero so that steps are
        // included and requeued jobs show only their latest run, as in sacct's defaults

        Self { cond }
    }
}

impl Drop for JobCond {
    fn drop(&mut self) {
        if !self.cond.cluster_list.is_null() {
            unsafe { slurm_list_destroy(self.cond.cluster_list) }
            self.cond.cluster_list = std::ptr::null_mut();
        }
    }
}

/// The job records returned by the accounting database
struct SlurmJobList {
    ptr: *mut xlist,
}

impl SlurmJobList {
    fn new(db_conn: &mut DbConn, job_cond: &mut JobCond) -> Result<Self, SacctError> {
        let ptr = unsafe { slurmdb_jobs_get(db_conn.as_mut_ptr(), &mut job_cond.cond) };
        if ptr.is_null() {
            Err(SacctError::JobListNull)
        } else {
            Ok(Self { ptr })
        }
    }

    /// Walks the records; the borrow keeps the list alive for as long as the iterator
    fn iter(&self) -> SlurmIterator<'_> {
        unsafe { SlurmIterator::new(self.ptr) }
    }
}

impl Drop for SlurmJobList {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { slurm_list_destroy(self.ptr) }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// The command line a job or one of its steps was submitted with, as reported by
/// `sacct --format JobIDRaw,SubmitLine`
#[derive(Debug, Clone)]
pub struct JobSubmitLine {
    pub job_id: u32,
    /// `None` for the job itself, which sacct prints on its own row ahead of its steps
    pub step_id: Option<StepId>,
    /// Empty where Slurm recorded none, e.g. for the batch and extern steps
    pub submit_line: String,
}

impl JobSubmitLine {
    /// The ID as sacct prints it in the JobIDRaw column, e.g. `1234` or `1234.batch`
    pub fn job_id_raw(&self) -> String {
        match self.step_id {
            Some(step_id) => format!("{}.{}", self.job_id, step_id),
            None => self.job_id.to_string(),
        }
    }
}

/// Reads one job record, followed by each of its steps, in the order sacct prints them
unsafe fn read_job(rec: *const slurmdb_job_rec_t) -> Vec<JobSubmitLine> {
    let job = unsafe { &*rec };

    let mut lines = vec![JobSubmitLine {
        job_id: job.jobid,
        step_id: None,
        submit_line: unsafe { c_str_to_string(job.submit_line) },
    }];

    for node_ptr in unsafe { SlurmIterator::new(job.steps) } {
        let step = unsafe { &*(node_ptr as *const slurmdb_step_rec_t) };
        lines.push(JobSubmitLine {
            job_id: job.jobid,
            step_id: Some(StepId::from(step.step_id.step_id)),
            submit_line: unsafe { c_str_to_string(step.submit_line) },
        });
    }

    lines
}

/// sacct's default start time when given neither jobs nor states: midnight local time today
fn local_midnight() -> DateTime<Utc> {
    Local::now()
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .and_then(|midnight| midnight.and_local_timezone(Local).earliest())
        .map(|midnight| midnight.with_timezone(&Utc))
        // a zone whose clocks skip midnight for DST has none today; a day back covers it
        .unwrap_or_else(|| Utc::now() - Duration::days(1))
}

/// The submit lines of every job and step in the accounting database, the equivalent of
///
/// ```text
/// sacct --noheader --parsable2 --allusers --format JobIDRaw,SubmitLine
/// ```
///
/// `usage_start` and `usage_end` are sacct's `-S` and `-E`, and default as they do there: to
/// midnight local time today, and to now. Unless the caller is a Slurm operator or admin,
/// slurmdbd may still withhold other users' jobs under `PrivateData=jobs`, as it would
/// from sacct.
pub fn get_submit_lines(
    usage_start: Option<DateTime<Utc>>,
    usage_end: Option<DateTime<Utc>>,
) -> Result<Vec<JobSubmitLine>, SacctError> {
    let usage_start = usage_start.unwrap_or_else(local_midnight);
    let usage_end = usage_end.unwrap_or_else(Utc::now);

    let mut persist_flags: u16 = 0;
    let mut db_conn = slurmdb_connect(&mut persist_flags).map_err(|_| SacctError::DbConnError)?;

    let mut job_cond = JobCond::new(usage_start, usage_end);
    let job_list = SlurmJobList::new(&mut db_conn, &mut job_cond)?;

    Ok(job_list
        .iter()
        .flat_map(|node_ptr| unsafe { read_job(node_ptr as *const slurmdb_job_rec_t) })
        .collect())
}
