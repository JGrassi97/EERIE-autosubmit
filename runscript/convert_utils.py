import xarray as xr
import numpy as np

def add_specific_cloud_water_content(eerie_sfc: xr.Dataset, eerie_plev: xr.Dataset,
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

    def normalize_coords(ds):
        ren = {}
        if 'latitude' in ds.coords: ren['latitude'] = 'lat'
        if 'longitude' in ds.coords: ren['longitude'] = 'lon'
        if 'lev' in ds.coords: ren['lev'] = 'plev'
        if 'level' in ds.coords: ren['level'] = 'plev'
        return ds.rename(ren) if ren else ds

    eerie_sfc = normalize_coords(eerie_sfc)
    eerie_plev = normalize_coords(eerie_plev)

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