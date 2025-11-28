import xarray as xr
import numpy as np

def add_specific_cloud_water_content_uniform(eerie_sfc: xr.Dataset, eerie_plev: xr.Dataset,
                                     conserve_mass: bool = True) -> xr.Dataset:
    """
    Aggiunge a eerie_plev:
      - specific_cloud_liquid_water_content (clwc)
      - specific_cloud_ice_water_content    (ciwc)

    Se conserve_mass=True (default):
        clwc_k = tclw * g / sum(Δp)
        ciwc_k = tciw * g / sum(Δp)
    (uniforme per unità di massa, conserva la colonna)

    Se conserve_mass=False (sconsigliato):
        clwc_k = tclw * g / Δp_k
        ciwc_k = tciw * g / Δp_k
    (non conserva; può dare ordini di grandezza troppo grandi)
    """
    g = 9.80665  # m s^-2

    def normalize_names(ds):
        ren = {}
        if 'latitude' in ds.coords: ren['latitude'] = 'lat'
        if 'longitude' in ds.coords: ren['longitude'] = 'lon'
        if 'lev' in ds.coords: ren['lev'] = 'plev'
        if 'level' in ds.coords: ren['level'] = 'plev'
        if 'pressure_level' in ds.coords: ren['pressure_level'] = 'plev'
        if 'valid_time' in ds.coords: ren['valid_time'] = 'time'

        return ds.rename(ren) if ren else ds

    eerie_sfc = normalize_names(eerie_sfc)
    eerie_plev = normalize_names(eerie_plev)

    # Input
    tclw = eerie_sfc['tclw']  # kg m^-2
    tciw = eerie_sfc['tciw']  # kg m^-2
    Tref = eerie_plev['t']    # per shape (time, plev, lat, lon)
    p    = eerie_plev['plev'] # (plev,)

    # Allinea time/lat/lon
    Tref, tclw, tciw = xr.align(Tref, tclw, tciw, join='inner')

    # p in Pa (NumPy)
    units = (p.attrs.get('units','') or '').lower()
    if units in ['hpa','millibar','mb']:
        p_vals = p.values.astype('float64') * 100.0
    else:
        p_vals = p.values.astype('float64')

    # Δp per livello (NumPy, robusto)
    edges = np.empty(p_vals.size + 1, dtype='float64')
    edges[1:-1] = 0.5*(p_vals[:-1] + p_vals[1:])
    edges[0]    = p_vals[0]  + 0.5*(p_vals[0]  - p_vals[1])
    edges[-1]   = p_vals[-1] + 0.5*(p_vals[-1] - p_vals[-2])
    dp_vals = np.abs(np.diff(edges))  # Pa

    dp = xr.DataArray(dp_vals, coords={'plev': p}, dims=['plev'])

    # Broadcast
    dp_4d   = dp.broadcast_like(Tref)
    tclw_4d = tclw.broadcast_like(Tref)
    tciw_4d = tciw.broadcast_like(Tref)

    if conserve_mass:
        # somma verticale degli spessori
        sum_dp = dp_4d.isel(time=0, lat=0, lon=0)*0 + dp_4d  # ensure same dims
        sum_dp = sum_dp.sum('plev', keep_attrs=False)         # (time, lat, lon)
        sum_dp = sum_dp.broadcast_like(Tref)                  # 4D

        clwc = tclw_4d * g / sum_dp
        ciwc = tciw_4d * g / sum_dp
        method = 'Mass-conserving uniform mixing ratio: xiwc = tciw_or_tclw * g / sum(Δp)'
    else:
        clwc = tclw_4d * g / dp_4d
        ciwc = tciw_4d * g / dp_4d
        method = 'Non-conserving layer formula: xiwc = tciw_or_tclw * g / Δp_k (not recommended)'

    clwc.name = 'specific_cloud_liquid_water_content'
    ciwc.name = 'specific_cloud_ice_water_content'

    clwc.attrs.update({
        'long_name': 'Specific cloud liquid water content',
        'short_name': 'clwc',
        'units': 'kg kg-1',
        'method': method
    })
    ciwc.attrs.update({
        'long_name': 'Specific cloud ice water content',
        'short_name': 'ciwc',
        'units': 'kg kg-1',
        'method': method
    })

    return eerie_plev.assign(
        specific_cloud_liquid_water_content=clwc,
        specific_cloud_ice_water_content=ciwc
    )





def add_specific_cloud_water_content(
    eerie_sfc: xr.Dataset,
    eerie_plev: xr.Dataset,
    liquid_min_temp: float = 273.15 - 38.0,  # K, supercooled water limit (~-38°C)
    ice_max_temp: float = 273.15,            # K, upper ice limit (0°C)
) -> xr.Dataset:
    """
    Add to eerie_plev:
      - specific_cloud_liquid_water_content (clwc)
      - specific_cloud_ice_water_content    (ciwc)

    Physical method based on:
      - Cloud mask from critical RH (Salonen & Uppala 1991, in σ = p / psfc)
      - Liquid phase: modified adiabatic profile (Karstens-like) vs. z - z_base
      - Ice phase: exponential temperature weighting (Liou/Ou-style) × RH

    Column mass is conserved:
        TCCLW = Σ_k clwc_k Δp_k / g
        TCCIW = Σ_k ciwc_k Δp_k / g
    with Δp_k the pressure thickness of level k.

    Parameters
    ----------
    eerie_sfc : xr.Dataset
        Must contain:
            - tclw : Total column cloud liquid water (kg m^-2)
            - tciw : Total column cloud ice water   (kg m^-2)
        Ideally also:
            - sp / ps / surface_pressure (Pa or hPa) for σ (Salonen–Uppala).

    eerie_plev : xr.Dataset
        Must contain at least:
            - t      : temperature (K), dims (time, plev, lat, lon)
            - plev   : pressure (Pa or hPa)
            - RH     : relative humidity (e.g. 'r', 'rh', in fraction or %)
        Ideally:
            - geopotential (z or geopotential) for geometric height (m).

    Returns
    -------
    xr.Dataset
        eerie_plev with two new variables:
            - specific_cloud_liquid_water_content (kg kg-1)
            - specific_cloud_ice_water_content    (kg kg-1)
    """

    g = 9.80665
    R_d = 287.058

    # ------------------------------------------------------------------
    #  Helper: normalize coordinate names
    # ------------------------------------------------------------------
    def normalize_names(ds):
        ren = {}
        if 'latitude' in ds.coords: ren['latitude'] = 'lat'
        if 'longitude' in ds.coords: ren['longitude'] = 'lon'
        if 'lev' in ds.coords: ren['lev'] = 'plev'
        if 'level' in ds.coords: ren['level'] = 'plev'
        if 'pressure_level' in ds.coords: ren['pressure_level'] = 'plev'
        if 'valid_time' in ds.coords: ren['valid_time'] = 'time'
        return ds.rename(ren) if ren else ds

    eerie_sfc = normalize_names(eerie_sfc)
    eerie_plev = normalize_names(eerie_plev)

    # ------------------------------------------------------------------
    #  Essential input variables
    # ------------------------------------------------------------------
    tclw = eerie_sfc['tclw']  # kg m^-2
    tciw = eerie_sfc['tciw']  # kg m^-2

    T = eerie_plev['t']       # K, (time, plev, lat, lon)
    p = eerie_plev['plev']    # (plev,)

    # Relative humidity: try common names
    rh_var_name = None
    for cand in ['r', 'rh', 'relative_humidity', 'hur']:
        if cand in eerie_plev.data_vars:
            rh_var_name = cand
            break
    if rh_var_name is None:
        raise KeyError(
            "No relative humidity variable found in eerie_plev "
            "(expected one of: 'r', 'rh', 'relative_humidity', 'hur')."
        )

    RH = eerie_plev[rh_var_name]

    # Geopotential / height (optional but recommended)
    z = None
    geopot_name = None
    for cand in ['z', 'geopotential', 'gh']:
        if cand in eerie_plev.data_vars:
            geopot_name = cand
            break

    if geopot_name is not None:
        geo = eerie_plev[geopot_name]
        units_geo = (geo.attrs.get('units', '') or '').lower()
        # If units ~ m2/s2 → z = Φ / g, else assume meters
        if 'm2 s-2' in units_geo or 'm^2 s^-2' in units_geo or 'm2/s2' in units_geo:
            z = geo / g
        else:
            z = geo
    else:
        # Fallback: monotonic "pseudo-height" from pressure (relative shape only)
        # z_tilde ∝ -ln(p / p_max)
        units_p = (p.attrs.get('units', '') or '').lower()
        if units_p in ['hpa', 'millibar', 'mb']:
            p_vals = p.values.astype('float64') * 100.0
        else:
            p_vals = p.values.astype('float64')
        z_1d = -np.log(p_vals / np.max(p_vals))
        z = xr.DataArray(z_1d, coords={'plev': p}, dims=['plev'])

    # ------------------------------------------------------------------
    #  Align on (time, plev, lat, lon)
    # ------------------------------------------------------------------
    if set(z.dims) == {'plev'}:
        z = z.broadcast_like(T)

    T, RH, z, tclw, tciw = xr.align(T, RH, z, tclw, tciw, join='inner')

    # ------------------------------------------------------------------
    #  Pressure and dp for vertical sum (Pa)
    # ------------------------------------------------------------------
    units_p = (p.attrs.get('units', '') or '').lower()
    if units_p in ['hpa', 'millibar', 'mb']:
        p_vals = p.values.astype('float64') * 100.0
    else:
        p_vals = p.values.astype('float64')

    # Δp 1D
    edges = np.empty(p_vals.size + 1, dtype='float64')
    edges[1:-1] = 0.5 * (p_vals[:-1] + p_vals[1:])
    edges[0]    = p_vals[0]  + 0.5 * (p_vals[0]  - p_vals[1])
    edges[-1]   = p_vals[-1] + 0.5 * (p_vals[-1] - p_vals[-2])
    dp_vals = np.abs(np.diff(edges))  # Pa

    dp = xr.DataArray(dp_vals, coords={'plev': p}, dims=['plev'])
    dp_4d = dp.broadcast_like(T)

    # Σ Δp per column (time, lat, lon)
    sum_dp = dp_4d.sum('plev', keep_attrs=False)
    sum_dp_4d = sum_dp.broadcast_like(T)

    # ------------------------------------------------------------------
    #  Surface pressure for σ = p / psfc (Salonen–Uppala)
    # ------------------------------------------------------------------
    psfc = None
    for cand in ['sp', 'ps', 'surface_pressure', 'pressure']:
        if cand in eerie_sfc.data_vars:
            psfc = eerie_sfc[cand]
            break

    if psfc is not None:
        units_ps = (psfc.attrs.get('units', '') or '').lower()
        if units_ps in ['hpa', 'millibar', 'mb']:
            psfc = psfc * 100.0
    else:
        # Fallback: use maximum pressure in column
        p_4d = xr.DataArray(p_vals, coords={'plev': p}, dims=['plev']).broadcast_like(T)
        psfc = p_4d.max('plev')
    psfc_4d = psfc.broadcast_like(T)

    # ------------------------------------------------------------------
    #  Relative humidity in fraction [0–1]
    # ------------------------------------------------------------------
    rh_units = (RH.attrs.get('units', '') or '').lower()
    RH_frac = RH.astype('float64')

    if ('%' in rh_units) or ('percent' in rh_units):
        RH_frac = RH_frac / 100.0
    else:
        # If values are typically between 0 and 100, assume percent
        rh_max = float(RH_frac.max().values)
        if rh_max > 2.0:
            RH_frac = RH_frac / 100.0

    RH_frac = RH_frac.clip(0.0, 1.5)

    # ------------------------------------------------------------------
    #  Cloud mask: Salonen & Uppala (1991)
    #     RH_c(σ) = 1 - α σ (1 - σ) [1 + β (σ - 0.5)]
    # ------------------------------------------------------------------
    alpha = 1.0
    beta = np.sqrt(3.0)

    p_4d = xr.DataArray(p_vals, coords={'plev': p}, dims=['plev']).broadcast_like(T)
    sigma = p_4d / psfc_4d
    sigma = sigma.clip(0.0, 1.2)

    RH_crit = 1.0 - alpha * sigma * (1.0 - sigma) * (1.0 + beta * (sigma - 0.5))
    RH_crit = RH_crit.clip(0.0, 1.0)

    cloud_active = RH_frac > RH_crit

    # ------------------------------------------------------------------
    #  Phase masks
    # ------------------------------------------------------------------
    liquid_mask = cloud_active & (T >= liquid_min_temp)
    ice_mask    = cloud_active & (T <= ice_max_temp)

    # ------------------------------------------------------------------
    #  Liquid phase shape (Karstens-like)
    #      S_liq(z) ∝ (z - z_base) [1.239 - 0.145 ln(z - z_base)]
    # ------------------------------------------------------------------
    z_liq = z.where(liquid_mask)
    z_base = z_liq.min('plev', skipna=True)  # (time, lat, lon)
    z_base_4d = z_base.broadcast_like(T)

    z_rel = (z - z_base_4d).where(liquid_mask)

    # Avoid log(0): minimum vertical distance 1 m
    z_rel_pos = z_rel.clip(min=1.0)

    S_liq = z_rel_pos * (1.239 - 0.145 * np.log(z_rel_pos))
    # No negative weights
    S_liq = S_liq.clip(min=0.0)
    S_liq = S_liq.where(liquid_mask, 0.0).fillna(0.0)

    # ------------------------------------------------------------------
    #  Ice phase shape
    #      S_ice(T) ∝ exp(0.04 * T_C) * RH
    # ------------------------------------------------------------------
    T_C = T - 273.15
    S_ice = np.exp(0.04 * T_C) * RH_frac
    S_ice = S_ice.where(ice_mask, 0.0).fillna(0.0)

    # ------------------------------------------------------------------
    #  Normalization: mass conservation
    #      q_k = A * S_k
    #      TCCW = Σ q_k Δp_k / g  ⇒ A = TCCW * g / Σ (S_k Δp_k)
    # ------------------------------------------------------------------
    tclw_4d = tclw.broadcast_like(T)
    tciw_4d = tciw.broadcast_like(T)

    # LIQUID
    denom_liq = (S_liq * dp_4d).sum('plev', keep_attrs=False)  # (time, lat, lon)
    denom_liq_4d = denom_liq.broadcast_like(T)

    A_liq = xr.where(
        denom_liq_4d > 0.0,
        tclw_4d * g / denom_liq_4d,
        0.0
    )
    clwc_shape = A_liq * S_liq  # kg kg-1

    # Fallback: if TCCLW>0 but no valid S_liq profile, use uniform mixing ratio
    liquid_need_fallback = (denom_liq_4d <= 0.0) & (tclw_4d > 0.0)
    uniform_clwc = xr.where(
        sum_dp_4d > 0.0,
        tclw_4d * g / sum_dp_4d,
        0.0
    )
    clwc = xr.where(liquid_need_fallback, uniform_clwc, clwc_shape)

    # ICE
    denom_ice = (S_ice * dp_4d).sum('plev', keep_attrs=False)
    denom_ice_4d = denom_ice.broadcast_like(T)

    A_ice = xr.where(
        denom_ice_4d > 0.0,
        tciw_4d * g / denom_ice_4d,
        0.0
    )
    ciwc_shape = A_ice * S_ice  # kg kg-1

    ice_need_fallback = (denom_ice_4d <= 0.0) & (tciw_4d > 0.0)
    uniform_ciwc = xr.where(
        sum_dp_4d > 0.0,
        tciw_4d * g / sum_dp_4d,
        0.0
    )
    ciwc = xr.where(ice_need_fallback, uniform_ciwc, ciwc_shape)

    # ------------------------------------------------------------------
    #  Metadata and output
    # ------------------------------------------------------------------
    clwc.name = 'specific_cloud_liquid_water_content'
    ciwc.name = 'specific_cloud_ice_water_content'

    method_liq = (
        "Mass-conserving diagnostic profile: "
        "clwc_k = A_liq * S_liq(z), with S_liq ∝ (z - z_base)[1.239 - 0.145 ln(z - z_base)] "
        "for levels with RH>RH_c(σ) and T >= {:.1f}K (Salonen–Uppala + Karstens-like). "
        "Columns without physical solution use a uniform mixing ratio."
    ).format(liquid_min_temp)

    method_ice = (
        "Mass-conserving diagnostic profile: "
        "ciwc_k = A_ice * S_ice(T,RH), with S_ice ∝ exp(0.04 T_C) * RH "
        "for levels with RH>RH_c(σ) and T <= {:.1f}K (Salonen–Uppala + thermal weighting). "
        "Columns without physical solution use a uniform mixing ratio."
    ).format(ice_max_temp)

    clwc.attrs.update({
        'long_name': 'Specific cloud liquid water content',
        'short_name': 'clwc',
        'units': 'kg kg-1',
        'method': method_liq
    })
    ciwc.attrs.update({
        'long_name': 'Specific cloud ice water content',
        'short_name': 'ciwc',
        'units': 'kg kg-1',
        'method': method_ice
    })

    clwc = clwc.clip(min=0.0)
    ciwc = ciwc.clip(min=0.0)

    return eerie_plev.assign(
        specific_cloud_liquid_water_content=clwc,
        specific_cloud_ice_water_content=ciwc
    )