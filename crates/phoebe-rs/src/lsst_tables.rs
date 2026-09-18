//! LSST ugrizy throughput, read from the sncosmo bandpass files.
//!
//! The six files in `data/lsst/` are copied verbatim from
//! <https://github.com/sncosmo/sncosmo.github.io/tree/master/data/bandpasses/lsst>,
//! which publishes the LSST `syseng_throughputs` v1.1 totals: telescope,
//! camera, filter and atmosphere together, which is what a photometric
//! measurement actually sees. Each file carries its own provenance header,
//! then two columns, wavelength in nm and throughput between 0 and 1, on a
//! uniform 0.1 nm grid from 300 to 1150 nm.
//!
//! The files are embedded with `include_str!`, so they are parsed once on
//! first use and nothing is read from disk at run time: the crate stays a
//! single artefact with no data path to configure. Keeping them in sncosmo's
//! own format means they can be re-copied from upstream without a conversion
//! step in between.

use std::sync::OnceLock;

/// First wavelength of the tabulated grid, nm.
pub const LAM_LO_NM: f64 = 300.0;
/// Grid spacing, nm.
pub const LAM_STEP_NM: f64 = 0.1;
/// Number of grid points, 300 to 1150 nm inclusive.
pub const NLAM: usize = 8501;

/// The vendored files, in the ugrizy order used everywhere else.
static FILES: [&str; 6] = [
    include_str!("../data/lsst/total_u.dat"),
    include_str!("../data/lsst/total_g.dat"),
    include_str!("../data/lsst/total_r.dat"),
    include_str!("../data/lsst/total_i.dat"),
    include_str!("../data/lsst/total_z.dat"),
    include_str!("../data/lsst/total_y.dat"),
];

/// Band names, for error messages.
static NAMES: [&str; 6] = ["u", "g", "r", "i", "z", "y"];

/// Parse one sncosmo bandpass file into throughput on the declared grid.
///
/// The wavelength column is checked rather than ignored: the grid is a
/// compile-time constant here, so a file on a different one would otherwise
/// shift every wavelength silently.
fn parse(src: &str, name: &str) -> Vec<f64> {
    let mut trans = Vec::with_capacity(NLAM);
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split_whitespace();
        let lam: f64 = cols
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("lsst {name}: bad wavelength in {line:?}"));
        let t: f64 = cols
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("lsst {name}: bad throughput in {line:?}"));
        let want = LAM_LO_NM + LAM_STEP_NM * trans.len() as f64;
        assert!(
            (lam - want).abs() < 1e-6,
            "lsst {name}: expected {want} nm at row {}, found {lam}",
            trans.len()
        );
        trans.push(t);
    }
    assert_eq!(trans.len(), NLAM, "lsst {name}: expected {NLAM} rows");
    trans
}

/// All six curves, parsed once.
fn curves() -> &'static [Vec<f64>; 6] {
    static CURVES: OnceLock<[Vec<f64>; 6]> = OnceLock::new();
    CURVES.get_or_init(|| {
        std::array::from_fn(|i| parse(FILES[i], NAMES[i]))
    })
}

/// Throughput of one LSST band on the uniform grid, in ugrizy order.
pub fn lsst_curve(index: usize) -> &'static [f64] {
    &curves()[index]
}

/// Pivot wavelength of each band, nm, in ugrizy order.
///
/// Computed from the curves themselves rather than tabulated separately, so
/// the effective-wavelength mode and the integrated mode can never describe
/// two different filters. Trapezoidal, with the usual definition
/// `lam_p^2 = int S lam dlam / int (S / lam) dlam`.
fn pivots() -> &'static [f64; 6] {
    static PIVOTS: OnceLock<[f64; 6]> = OnceLock::new();
    PIVOTS.get_or_init(|| {
        std::array::from_fn(|i| {
            let s = lsst_curve(i);
            let lam = |k: usize| LAM_LO_NM + LAM_STEP_NM * k as f64;
            let (mut num, mut den) = (0.0, 0.0);
            for k in 0..NLAM - 1 {
                let (l0, l1) = (lam(k), lam(k + 1));
                let (s0, s1) = (s[k], s[k + 1]);
                let h = 0.5 * (l1 - l0);
                num += h * (s0 * l0 + s1 * l1);
                den += h * (s0 / l0 + s1 / l1);
            }
            (num / den).sqrt()
        })
    })
}

/// Pivot wavelength of one LSST band, nm, in ugrizy order.
pub fn lsst_pivot_nm(index: usize) -> f64 {
    pivots()[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_band_parses_onto_the_declared_grid() {
        // parse() asserts the grid, so this also pins the file format: if a
        // re-copy from sncosmo changed it, the failure lands here and not in
        // the middle of a light curve.
        for i in 0..6 {
            let c = lsst_curve(i);
            assert_eq!(c.len(), NLAM, "band {}", NAMES[i]);
            assert!(c.iter().all(|t| (0.0..=1.0).contains(t)),
                    "band {} has throughput outside 0 to 1", NAMES[i]);
            assert!(c.iter().any(|&t| t > 0.1),
                    "band {} never rises above 0.1", NAMES[i]);
        }
    }

    #[test]
    fn pivot_wavelengths_are_ordered_and_in_band() {
        // ugrizy run blue to red, and each pivot has to sit inside its own
        // filter rather than merely between 300 and 1150 nm.
        let p = pivots();
        for i in 0..5 {
            assert!(p[i] < p[i + 1], "pivots out of order: {:?}", p);
        }
        for i in 0..6 {
            let c = lsst_curve(i);
            let peak = c.iter().cloned().fold(0.0_f64, f64::max);
            let k = ((p[i] - LAM_LO_NM) / LAM_STEP_NM).round() as usize;
            assert!(c[k] > 0.5 * peak,
                    "band {} pivot {} nm sits off the filter", NAMES[i], p[i]);
        }
    }
}
