//! Generate a multiband eclipsing-binary template bank with phoebe-rs.
//!
//! One example covering everything the crate offers: contact, detached and
//! semidetached systems, all six LSST bands from a single Roche mesh, and both
//! optional behaviours.
//!
//! Usage:
//!
//!     genbank <specs.csv> [nphase]
//!
//! The CSV names its columns, order does not matter:
//!
//!     family,q,inclination,temperature_ratio,primary_temperature,
//!     r1,r2,companion_fill,filling_component
//!
//! `family` is contact, detached or semidetached, and it decides how r1 and r2
//! are read:
//!
//! * `contact`      — r1 and r2 are fillout values, used as given. Both stars
//!                    overflow, and the surface potential interpolates L1 to L2.
//! * `detached`     — r1 and r2 are FRACTIONAL RADII in units of the separation
//!                    and are converted. `fillout` is a fraction of the distance
//!                    to L1, not a radius, so a radius passed straight in makes
//!                    a star that is far too small.
//! * `semidetached` — the component named by `filling_component` fills its lobe
//!                    (fillout 1.0) and the other sits at `companion_fill`.
//!
//! Two environment variables select the optional behaviours, both defaulting to
//! what the crate did before they existed:
//!
//! * `PHOEBE_RS_BAND_MODE=integrated` integrates Planck against the tabulated
//!   LSST throughput instead of evaluating it once at the band's pivot
//!   wavelength. Worth about 0.0015 mag and costs nothing measurable, because
//!   SBTable precomputes 161 temperatures once per band.
//! * `PHOEBE_RS_ECLIPSE_ACC=scaled` ties the line-of-sight eclipse tolerance to
//!   the eclipsing star's size. The fixed default is wider than the whole star
//!   once its radius falls below about 0.025, and the eclipse then disappears
//!   rather than becoming imprecise: 0.033 mag where it should be 0.546.
//!
//! Output is six rows per spec, one per band in ugrizy order, each of `nphase`
//! space-separated fluxes. A spec that cannot be built writes six rows of NaN,
//! so rows stay aligned with the input, and the reason goes to stderr.
//!
//! Eccentricity is not a parameter phoebe-rs has; eccentric systems cannot be
//! generated here.
use std::env;
use std::fs;

use lcurve_roche::lagrange::{xl11, xl12};
use phoebe_rs::lightcurve::compute_lightcurve_multiband;
use phoebe_rs::params::EBParams;
use phoebe_rs::passband::Passband;
use phoebe_rs::EBType;

/// Convert a fractional radius, in units of the separation, to `fillout`.
///
/// `ref_sphere` places the surface at `fillout * rl`, where rl is the distance
/// from the star's centre to L1. Synchronous rotation, spin = 1.
fn fillout_for_radius(q: f64, r: f64, primary: bool) -> f64 {
    let rl = if primary {
        xl11(q, 1.0).unwrap_or(f64::NAN)
    } else {
        1.0 - xl12(q, 1.0).unwrap_or(f64::NAN)
    };
    if !rl.is_finite() || rl <= 0.0 {
        return f64::NAN;
    }
    r / rl
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: genbank <specs.csv> [nphase]");
        std::process::exit(2);
    }
    let nph: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(513);
    let text = fs::read_to_string(&args[1]).expect("cannot read specs");

    let bands = [
        Passband::LsstU, Passband::LsstG, Passband::LsstR,
        Passband::LsstI, Passband::LsstZ, Passband::LsstY,
    ];
    // phase 1 repeats phase 0, so a consumer can fold without a special case
    let phases: Vec<f64> = (0..nph).map(|i| i as f64 / (nph - 1) as f64).collect();
    let nan_row: String = vec!["NaN"; nph].join(" ");

    eprintln!(
        "band mode {}, eclipse accuracy {}, {} phases",
        env::var("PHOEBE_RS_BAND_MODE").unwrap_or_else(|_| "effective (default)".into()),
        env::var("PHOEBE_RS_ECLIPSE_ACC").unwrap_or_else(|_| "fixed (default)".into()),
        nph
    );

    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("specs.csv is missing column {name}"))
    };
    let (i_fam, i_q, i_inc, i_tr, i_t1, i_r1, i_r2, i_cf, i_fc) = (
        idx("family"), idx("q"), idx("inclination"), idx("temperature_ratio"),
        idx("primary_temperature"), idx("r1"), idx("r2"),
        idx("companion_fill"), idx("filling_component"),
    );

    let (mut nok, mut nfail) = (0usize, 0usize);
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        let num = |i: usize| f[i].parse::<f64>().unwrap_or(f64::NAN);
        let (family, fill_comp) = (f[i_fam], f[i_fc]);
        let (q, inc, tr, t1) = (num(i_q), num(i_inc), num(i_tr), num(i_t1));
        let (r1, r2, cfill) = (num(i_r1), num(i_r2), num(i_cf));

        let mut p = match family {
            "contact" => {
                let mut p = EBParams::contact(q, inc);
                p.eb_type = EBType::Contact;
                p.fillout1 = r1;
                p.fillout2 = r2;
                p
            }
            "detached" => {
                let f1 = fillout_for_radius(q, r1, true);
                let f2 = fillout_for_radius(q, r2, false);
                let mut p = EBParams::detached(q, inc, f1, f2);
                p.eb_type = EBType::Detached;
                p
            }
            "semidetached" => {
                let mut p = EBParams::detached(q, inc, 0.5, 0.5);
                p.eb_type = EBType::SemiDetached;
                if fill_comp == "primary" {
                    p.fillout1 = 1.0;
                    p.fillout2 = cfill;
                } else {
                    p.fillout2 = 1.0;
                    p.fillout1 = cfill;
                }
                p
            }
            other => {
                eprintln!("row {n}: unsupported family {other}");
                for _ in 0..bands.len() {
                    println!("{nan_row}");
                }
                nfail += 1;
                continue;
            }
        };
        p.t_eff1 = t1;
        p.t_eff2 = t1 * tr;
        // ld1 and ld2 are ignored on the multiband path: each band uses its own
        // default_ld, which is what makes the six curves differ in shape and not
        // only in amplitude
        p.ld1 = 0.5;
        p.ld2 = 0.5;
        // a small star needs a finer surface mesh to be resolved at all
        p.n_grid = match family {
            "contact" => 20,
            _ => ((2.0_f64 / r1.min(r2).max(1e-3)).round() as usize).clamp(20, 400),
        };

        match compute_lightcurve_multiband(&p, &phases, &bands) {
            Ok(lcs) => {
                for lc in lcs.iter() {
                    let row: Vec<String> =
                        lc.flux.iter().map(|x| format!("{x:.6}")).collect();
                    println!("{}", row.join(" "));
                }
                nok += 1;
            }
            Err(e) => {
                eprintln!(
                    "row {n}: family={family} q={q} inc={inc} r1={r1} r2={r2} failed: {e:?}"
                );
                for _ in 0..bands.len() {
                    println!("{nan_row}");
                }
                nfail += 1;
            }
        }
    }
    eprintln!("generated {nok} templates, {nfail} failed");
}
