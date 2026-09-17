//! Light curve computation for eclipsing binaries.

use lcurve_roche::Star;
use crate::geometry::{build_mesh, build_overcontact_mesh, build_overcontact_mesh_marching};
use crate::flux::{line_of_sight, star_flux};
use crate::passband::{SBTable, Passband, BandMode};
use crate::{EBParams, EBType, PhoebeError};

/// Result of light curve computation.
#[derive(Debug)]
pub struct LightCurve {
    pub phases: Vec<f64>,
    pub flux: Vec<f64>,
    pub flux1: Vec<f64>,
    pub flux2: Vec<f64>,
}

/// Compute a synthetic eclipsing binary light curve.

/// The LSST band treatment, from PHOEBE_RS_BAND_MODE if it is set.
///
/// Defaults to one Planck evaluation at each band's pivot wavelength, which is
/// what the model did before the tables were added. Set the variable to
/// "integrated" to integrate Planck against the tabulated throughput instead.
fn band_mode() -> BandMode {
    std::env::var("PHOEBE_RS_BAND_MODE").ok()
        .and_then(|v| BandMode::from_str(&v))
        .unwrap_or_default()
}

pub fn compute_lightcurve(
    params: &EBParams,
    phases: &[f64],
) -> Result<LightCurve, PhoebeError> {
    params.validate()?;

    let q = params.q;
    let incl_rad = params.inclination.to_radians();

    // Build meshes depending on system type
    let (mesh1, mesh2) = match params.eb_type {
        EBType::Contact => {
            let oc_fillout = params.fillout1;
            // Use lat-lon grid with increased resolution (2x) for overcontact
            // to ensure good neck coverage
            build_overcontact_mesh(q, oc_fillout, params.n_grid * 2)?
        },
        _ => {
            // Detached or semi-detached: separate meshes
            let m1 = build_mesh(q, Star::Primary, params.fillout1, params.n_grid)?;
            let m2 = build_mesh(q, Star::Secondary, params.fillout2, params.n_grid)?;
            (m1, m2)
        },
    };

    let gd_exp = if params.t_eff1 > 7500.0 { 0.25 } else { 0.08 };
    let albedo = if params.t_eff1 > 7500.0 { 1.0 } else { 0.5 };

    // Build passband surface brightness lookup table
    let sb_table = SBTable::with_mode(params.passband, band_mode());

    let mut flux_arr = Vec::with_capacity(phases.len());
    let mut flux1_arr = Vec::with_capacity(phases.len());
    let mut flux2_arr = Vec::with_capacity(phases.len());

    // Normalisation at quadrature
    let los_quad = line_of_sight(0.25, incl_rad);

    // For eclipse checking, use the other star's filling factor
    let ff1 = match params.eb_type {
        EBType::Contact => 1.0,  // Roche-filling for eclipse check
        _ => params.fillout1,
    };
    let ff2 = match params.eb_type {
        EBType::Contact => 1.0,
        _ => params.fillout2,
    };

    let f1_quad = star_flux(&mesh1, &los_quad, q, Star::Secondary, ff2,
                            params.t_eff1, params.t_eff2, params.ld1, gd_exp, albedo, &sb_table);
    let f2_quad = star_flux(&mesh2, &los_quad, q, Star::Primary, ff1,
                            params.t_eff2, params.t_eff1, params.ld2, gd_exp, albedo, &sb_table);
    let f_norm = f1_quad + f2_quad;

    if f_norm <= 0.0 {
        return Err(PhoebeError::InvalidParams("zero flux at quadrature".to_string()));
    }

    for &phase in phases {
        let ph = phase + params.phi0;
        let los = line_of_sight(ph, incl_rad);

        let f1 = star_flux(&mesh1, &los, q, Star::Secondary, ff2,
                           params.t_eff1, params.t_eff2, params.ld1, gd_exp, albedo, &sb_table);
        let f2 = star_flux(&mesh2, &los, q, Star::Primary, ff1,
                           params.t_eff2, params.t_eff1, params.ld2, gd_exp, albedo, &sb_table);

        let total = (f1 + f2) / f_norm;
        let total_with_l3 = total * (1.0 - params.l3) + params.l3;

        flux_arr.push(total_with_l3);
        flux1_arr.push(f1 / f_norm);
        flux2_arr.push(f2 / f_norm);
    }

    Ok(LightCurve {
        phases: phases.to_vec(),
        flux: flux_arr,
        flux1: flux1_arr,
        flux2: flux2_arr,
    })
}

/// Compute light curves in several passbands at once.
///
/// The Roche mesh depends only on `q`, the filling factors and `n_grid`, so it is
/// built once and reused for every band. Only the surface-brightness table and the
/// limb-darkening coefficient vary between bands.
pub fn compute_lightcurve_multiband(
    params: &EBParams,
    phases: &[f64],
    bands: &[Passband],
) -> Result<Vec<LightCurve>, PhoebeError> {
    params.validate()?;

    let q = params.q;
    let incl_rad = params.inclination.to_radians();

    let (mesh1, mesh2) = match params.eb_type {
        EBType::Contact => {
            build_overcontact_mesh(q, params.fillout1, params.n_grid * 2)?
        },
        _ => {
            let m1 = build_mesh(q, Star::Primary, params.fillout1, params.n_grid)?;
            let m2 = build_mesh(q, Star::Secondary, params.fillout2, params.n_grid)?;
            (m1, m2)
        },
    };

    let gd_exp = if params.t_eff1 > 7500.0 { 0.25 } else { 0.08 };
    let albedo = if params.t_eff1 > 7500.0 { 1.0 } else { 0.5 };

    let ff1 = match params.eb_type { EBType::Contact => 1.0, _ => params.fillout1 };
    let ff2 = match params.eb_type { EBType::Contact => 1.0, _ => params.fillout2 };

    let los_quad = line_of_sight(0.25, incl_rad);
    let mut out = Vec::with_capacity(bands.len());

    for &pb in bands {
        let sb_table = SBTable::with_mode(pb, band_mode());
        let ld = pb.default_ld();

        let f1_quad = star_flux(&mesh1, &los_quad, q, Star::Secondary, ff2,
                                params.t_eff1, params.t_eff2, ld, gd_exp, albedo, &sb_table);
        let f2_quad = star_flux(&mesh2, &los_quad, q, Star::Primary, ff1,
                                params.t_eff2, params.t_eff1, ld, gd_exp, albedo, &sb_table);
        let f_norm = f1_quad + f2_quad;
        if f_norm <= 0.0 {
            return Err(PhoebeError::InvalidParams("zero flux at quadrature".to_string()));
        }

        let mut flux_arr = Vec::with_capacity(phases.len());
        let mut flux1_arr = Vec::with_capacity(phases.len());
        let mut flux2_arr = Vec::with_capacity(phases.len());

        for &phase in phases {
            let ph = phase + params.phi0;
            let los = line_of_sight(ph, incl_rad);
            let f1 = star_flux(&mesh1, &los, q, Star::Secondary, ff2,
                               params.t_eff1, params.t_eff2, ld, gd_exp, albedo, &sb_table);
            let f2 = star_flux(&mesh2, &los, q, Star::Primary, ff1,
                               params.t_eff2, params.t_eff1, ld, gd_exp, albedo, &sb_table);
            let total = (f1 + f2) / f_norm;
            flux_arr.push(total * (1.0 - params.l3) + params.l3);
            flux1_arr.push(f1 / f_norm);
            flux2_arr.push(f2 / f_norm);
        }

        out.push(LightCurve {
            phases: phases.to_vec(),
            flux: flux_arr,
            flux1: flux1_arr,
            flux2: flux2_arr,
        });
    }

    Ok(out)
}
