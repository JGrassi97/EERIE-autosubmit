#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import xarray as xr
import numpy as np
from convert_utils import add_specific_cloud_water_content


def main():
    parser = argparse.ArgumentParser(
        description="Prepare input layer by merging pressure-level and surface datasets."
    )
    parser.add_argument("--pl_path", required=True, help="Path to pressure-level NetCDF file")
    parser.add_argument("--sfc_path", required=True, help="Path to surface NetCDF file")
    parser.add_argument("--out_path", required=True, help="Output path for merged dataset")

    args = parser.parse_args()

    # 1. Load datasets
    eerie_pl = xr.open_dataset(args.pl_path)
    eerie_sfc = xr.open_dataset(args.sfc_path)

    # 2. Compute specific cloud water and ice content
    eerie_plev_out = add_specific_cloud_water_content(eerie_sfc, eerie_pl)

    # 3. Merge and rename variables
    eerie = xr.merge([eerie_plev_out, eerie_sfc], compat="override").rename({
        "plev": "level",
        "z": "geopotential",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "q": "specific_humidity",
        "sst": "sea_surface_temperature",
        "ci": "sea_ice_cover",
    }).rename({
        "lon": "longitude",
        "lat": "latitude",
    })

    # 4. Drop redundant total-column variables
    for var in ["tclw", "tciw"]:
        if var in eerie:
            eerie = eerie.drop_vars(var)

    # 5. Set vertical coordinate
    eerie = eerie.assign_coords({
        "level": np.array([
             1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
           125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
           550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
           925,  950,  975, 1000
        ], dtype=np.int32)
    })

    # 6. Save to output path
    eerie.to_netcdf(args.out_path)


if __name__ == "__main__":
    main()