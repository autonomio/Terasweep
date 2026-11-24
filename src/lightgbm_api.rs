use std::error::Error;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::ptr;

use crate::{lightgbm::TrainConfig, Matrix};

type Handle = *mut c_void;
type Free = unsafe extern "C" fn(Handle) -> c_int;
type LastError = unsafe extern "C" fn() -> *const c_char;
type DatasetCreate = unsafe extern "C" fn(
    *const c_void,
    c_int,
    i32,
    i32,
    c_int,
    *const c_char,
    Handle,
    *mut Handle,
) -> c_int;
type DatasetSetField =
    unsafe extern "C" fn(Handle, *const c_char, *const c_void, c_int, c_int) -> c_int;
type BoosterCreate = unsafe extern "C" fn(Handle, *const c_char, *mut Handle) -> c_int;
type BoosterUpdate = unsafe extern "C" fn(Handle, *mut c_int) -> c_int;
type BoosterPredict = unsafe extern "C" fn(
    Handle,
    *const c_void,
    c_int,
    i32,
    i32,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_char,
    *mut i64,
    *mut f64,
) -> c_int;

const DTYPE_FLOAT32: c_int = 0;
const DTYPE_FLOAT64: c_int = 1;
const PREDICT_NORMAL: c_int = 0;
const RTLD_NOW: c_int = 2;

#[cfg_attr(target_os = "linux", link(name = "dl"))]
extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> Handle;
    fn dlsym(handle: Handle, symbol: *const c_char) -> Handle;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: Handle) -> c_int;
}

struct Library(Handle);

impl Library {
    fn open() -> Result<Self, Box<dyn Error>> {
        let name = if cfg!(target_os = "macos") {
            "lib_lightgbm.dylib"
        } else {
            "lib_lightgbm.so"
        };
        let path = std::env::var_os("TERASWEEP_LIGHTGBM_LIB").unwrap_or_else(|| name.into());
        let c_path = CString::new(path.as_os_str().as_bytes())?;
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            let details = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy();
            return Err(format!(
                "Cannot load LightGBM at {}. Set TERASWEEP_LIGHTGBM_LIB to the official shared library. {details}",
                path.to_string_lossy()
            ).into());
        }
        Ok(Self(handle))
    }

    fn symbol<T: Copy>(&self, name: &str) -> Result<T, Box<dyn Error>> {
        let c_name = CString::new(name)?;
        let address = unsafe { dlsym(self.0, c_name.as_ptr()) };
        if address.is_null() {
            return Err(format!("LightGBM shared library is missing {name}").into());
        }
        // POSIX dlsym returns an address compatible with the named function's C ABI.
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe { dlclose(self.0) };
    }
}

struct OwnedHandle(Handle, Free);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        unsafe { (self.1)(self.0) };
    }
}

pub struct LightGbm {
    _library: Library,
    last_error: LastError,
    dataset_create: DatasetCreate,
    dataset_set_field: DatasetSetField,
    dataset_free: Free,
    booster_create: BoosterCreate,
    booster_update: BoosterUpdate,
    booster_predict: BoosterPredict,
    booster_free: Free,
}

impl LightGbm {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let library = Library::open()?;
        Ok(Self {
            last_error: library.symbol("LGBM_GetLastError")?,
            dataset_create: library.symbol("LGBM_DatasetCreateFromMat")?,
            dataset_set_field: library.symbol("LGBM_DatasetSetField")?,
            dataset_free: library.symbol("LGBM_DatasetFree")?,
            booster_create: library.symbol("LGBM_BoosterCreate")?,
            booster_update: library.symbol("LGBM_BoosterUpdateOneIter")?,
            booster_predict: library.symbol("LGBM_BoosterPredictForMat")?,
            booster_free: library.symbol("LGBM_BoosterFree")?,
            _library: library,
        })
    }

    fn check(&self, status: c_int) -> Result<(), Box<dyn Error>> {
        if status != 0 {
            let error = unsafe { CStr::from_ptr((self.last_error)()) };
            return Err(format!("LightGBM: {}", error.to_string_lossy()).into());
        }
        Ok(())
    }

    pub fn fit_predict(
        &self,
        train: &Matrix,
        labels: &[f32],
        test: &Matrix,
        config: TrainConfig,
        seed: u64,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        if train.cols != test.cols
            || train.rows.checked_mul(train.cols) != Some(train.x.len())
            || test.rows.checked_mul(test.cols) != Some(test.x.len())
            || labels.len() != train.rows
        {
            return Err("Invalid matrix or label dimensions for LightGBM".into());
        }
        let train_rows = i32::try_from(train.rows)?;
        let test_rows = i32::try_from(test.rows)?;
        let cols = i32::try_from(train.cols)?;
        let iterations = i32::try_from(config.num_trees)?;
        let params = CString::new(format!(
            "objective=regression boosting=gbdt learning_rate={} num_iterations={iterations} \
             num_leaves={} max_bin={} lambda_l2={} min_data_in_leaf={} seed={} \
             num_threads=1 deterministic=true force_col_wise=true verbosity=-1",
            config.learning_rate,
            config.num_leaves,
            config.max_bins,
            config.lambda_l2,
            config.min_data_in_leaf,
            seed & i32::MAX as u64,
        ))?;
        let mut dataset = ptr::null_mut();
        // Both matrices are contiguous row-major f64, and LightGBM labels are f32.
        self.check(unsafe {
            (self.dataset_create)(
                train.x.as_ptr().cast(),
                DTYPE_FLOAT64,
                train_rows,
                cols,
                1,
                params.as_ptr(),
                ptr::null_mut(),
                &mut dataset,
            )
        })?;
        let dataset = OwnedHandle(dataset, self.dataset_free);
        self.check(unsafe {
            (self.dataset_set_field)(
                dataset.0,
                b"label\0".as_ptr().cast(),
                labels.as_ptr().cast(),
                train_rows,
                DTYPE_FLOAT32,
            )
        })?;
        let mut booster = ptr::null_mut();
        self.check(unsafe { (self.booster_create)(dataset.0, params.as_ptr(), &mut booster) })?;
        let booster = OwnedHandle(booster, self.booster_free);
        for _ in 0..iterations {
            let mut finished = 0;
            self.check(unsafe { (self.booster_update)(booster.0, &mut finished) })?;
            if finished != 0 {
                break;
            }
        }

        let mut predictions = vec![0.0; test.rows];
        let mut count = 0;
        self.check(unsafe {
            (self.booster_predict)(
                booster.0,
                test.x.as_ptr().cast(),
                DTYPE_FLOAT64,
                test_rows,
                cols,
                1,
                PREDICT_NORMAL,
                0,
                -1,
                b"num_threads=1\0".as_ptr().cast(),
                &mut count,
                predictions.as_mut_ptr(),
            )
        })?;
        if count != i64::from(test_rows) {
            return Err(
                format!("LightGBM returned {count} predictions for {test_rows} rows").into(),
            );
        }
        Ok(predictions)
    }
}
