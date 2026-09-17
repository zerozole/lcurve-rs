//! phoebe-rs: Minimal eclipsing binary light curve synthesizer
//!
//! Focused on fast forward modelling of contact (W UMa) and detached
//! eclipsing binaries for mass ratio extraction from ZTF light curves.
//!
//! Uses the Roche geometry from lcurve-rs/roche crate.

pub mod params;
pub mod geometry;
pub mod flux;
pub mod lightcurve;
pub mod passband;
pub mod lsst_tables;
pub mod marching;
pub mod analytic;

pub use params::EBParams;
pub use passband::Passband;
pub use passband::BandMode;
pub use flux::EclipseAcc;
pub use lightcurve::compute_lightcurve;
pub use analytic::compute_analytic;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EBType {
    /// Contact binary (W UMa) - both stars fill Roche lobes
    Contact,
    /// Semi-detached - one star fills Roche lobe
    SemiDetached,
    /// Detached - neither star fills Roche lobe
    Detached,
}

#[derive(Debug)]
pub enum PhoebeError {
    Roche(lcurve_roche::RocheError),
    InvalidParams(String),
}

impl From<lcurve_roche::RocheError> for PhoebeError {
    fn from(e: lcurve_roche::RocheError) -> Self {
        PhoebeError::Roche(e)
    }
}
