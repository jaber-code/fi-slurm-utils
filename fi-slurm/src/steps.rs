use crate::jobs::JobState;
use crate::parser::parse_tres_str;
use crate::states::ShowFlags;
use crate::utils::{c_str_to_string, time_t_to_datetime};
use chrono::{DateTime, Utc};
use fi_slurm_sys::{
    NO_VAL, SLURM_BATCH_SCRIPT, SLURM_EXTERN_CONT, SLURM_INTERACTIVE_STEP, SLURM_PENDING_STEP,
    job_step_info_response_msg_t, job_step_info_t, slurm_free_job_step_info_response_msg,
    slurm_get_job_steps, slurm_step_id_t,
};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// We use this struct to manage the C-allocated memory,
/// automatically dropping it when it goes out of scope
pub struct RawSlurmStepInfo {
    ptr: *mut job_step_info_response_msg_t,
}

impl Drop for RawSlurmStepInfo {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                slurm_free_job_step_info_response_msg(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl RawSlurmStepInfo {
    /// Loads job step information from the Slurm controller.
    ///
    /// `job_id` and `step_id` narrow the request to one job or one step of it; Slurm reads
    /// `NO_VAL` in either as "all of them". This is the only function that directly calls
    /// the unsafe `slurm_get_job_steps` FFI function.
    pub fn load(job_id: u32, step_id: u32) -> Result<Self, String> {
        let mut step_info_msg_ptr: *mut job_step_info_response_msg_t = std::ptr::null_mut();

        // zeroed first so that any field this code does not name is left unset, as Slurm's
        // own initializer leaves it
        let mut step: slurm_step_id_t = unsafe { std::mem::zeroed() };
        step.job_id = job_id;
        step.step_id = step_id;
        // not a component of a heterogeneous step
        step.step_het_comp = NO_VAL;

        // ALL so that steps of jobs in hidden partitions are still returned
        let show_flags = ShowFlags::ALL;

        let return_code = unsafe {
            slurm_get_job_steps(&mut step, &mut step_info_msg_ptr, show_flags.bits())
        };

        if return_code != 0 || step_info_msg_ptr.is_null() {
            Err("Failed to load job step information from Slurm".to_string())
        } else {
            Ok(Self {
                ptr: step_info_msg_ptr,
            })
        }
    }

    /// Provides safe, read-only access to the step data as a Rust slice
    pub fn as_slice(&self) -> &[job_step_info_t] {
        if self.ptr.is_null() {
            return &[];
        }
        // This is `unsafe` because we are promising the compiler that the pointer
        // and job_step_count from the C library are valid
        unsafe {
            let msg = &*self.ptr;
            if msg.job_steps.is_null() {
                return &[];
            }
            std::slice::from_raw_parts(msg.job_steps, msg.job_step_count as usize)
        }
    }

    /// Consumes the wrapper to transform the raw C data into safe, owned `JobStep` values
    pub fn into_job_steps(self) -> Vec<JobStep> {
        self.as_slice().iter().map(JobStep::from_raw_binding).collect()
    }
}

/// Fetches job steps from Slurm and returns them as safe, owned Rust values.
///
/// `None` fetches the steps of every job; `Some(job_id)` only those of that job.
pub fn get_job_steps(job_id: Option<u32>) -> Result<Vec<JobStep>, String> {
    Ok(RawSlurmStepInfo::load(job_id.unwrap_or(NO_VAL), NO_VAL)?.into_job_steps())
}

/// Which step of a job this is. Slurm reserves a few step IDs at the top of the range for
/// steps it creates itself, which `scontrol` and `squeue` print by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepId {
    /// The step running the batch script
    Batch,
    /// The external step that adopts processes started outside of Slurm, e.g. over ssh
    Extern,
    /// The step `salloc` starts when `LaunchParameters=use_interactive_step` is set
    Interactive,
    /// A step that has been requested but not yet started
    Pending,
    /// An ordinary step, launched by `srun`
    Numbered(u32),
}

impl From<u32> for StepId {
    fn from(step_id: u32) -> Self {
        match step_id {
            SLURM_BATCH_SCRIPT => StepId::Batch,
            SLURM_EXTERN_CONT => StepId::Extern,
            SLURM_INTERACTIVE_STEP => StepId::Interactive,
            SLURM_PENDING_STEP => StepId::Pending,
            n => StepId::Numbered(n),
        }
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepId::Batch => write!(f, "batch"),
            StepId::Extern => write!(f, "extern"),
            StepId::Interactive => write!(f, "interactive"),
            StepId::Pending => write!(f, "TBD"),
            StepId::Numbered(n) => write!(f, "{n}"),
        }
    }
}

/// A safe, owned representation of a Slurm job step, holding a curated subset of the
/// fields of the raw C `job_step_info_t` struct
#[derive(Debug, Clone)]
pub struct JobStep {
    // Identification
    pub job_id: u32,
    pub step_id: StepId,
    pub name: String,
    pub user_id: u32,
    pub partition: String,

    // State and Time
    pub state: JobState,
    pub start_time: DateTime<Utc>,
    pub run_time: Duration,
    pub time_limit_minutes: u32,

    // Resource Allocation
    pub num_cpus: u32,
    pub num_tasks: u32,
    pub raw_hostlist: String,
    pub allocated_tres: HashMap<String, u64>,
}

impl JobStep {
    /// Creates a safe, owned Rust `JobStep` from a raw C `job_step_info_t` struct
    pub fn from_raw_binding(raw_step: &job_step_info_t) -> Self {
        JobStep {
            job_id: raw_step.step_id.job_id,
            step_id: StepId::from(raw_step.step_id.step_id),
            name: unsafe { c_str_to_string(raw_step.name) },
            user_id: raw_step.user_id,
            partition: unsafe { c_str_to_string(raw_step.partition) },
            state: JobState::from(raw_step.state),
            start_time: time_t_to_datetime(raw_step.start_time),
            // a step that has not started yet has no run time to report
            run_time: Duration::from_secs(raw_step.run_time.max(0) as u64),
            time_limit_minutes: raw_step.time_limit,
            num_cpus: raw_step.num_cpus,
            num_tasks: raw_step.num_tasks,
            raw_hostlist: unsafe { c_str_to_string(raw_step.nodes) },
            allocated_tres: unsafe { parse_tres_str(raw_step.tres_fmt_alloc_str) },
        }
    }

    /// The step's ID as Slurm prints it, e.g. `1234.batch` or `1234.0`
    pub fn full_id(&self) -> String {
        format!("{}.{}", self.job_id, self.step_id)
    }
}
