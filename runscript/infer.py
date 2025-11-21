#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import numpy as np
import xarray as xr
import gcsfs
import jax

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm


def load_model(model_name: str):
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{model_name}", "rb") as f:
        ckpt = pickle.load(f)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model


def build_regridder(src_ds: xr.Dataset, dst_grid) -> horizontal_interpolation.ConservativeRegridder:
    src_grid = spherical_harmonic.Grid(
        latitude_nodes=src_ds.sizes["latitude"],
        longitude_nodes=src_ds.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(src_ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(src_ds.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        src_grid, dst_grid, skipna=True
    )
    return regridder


def main():
    parser = argparse.ArgumentParser(
        description="Run NeuralGCM inference from prepared input layer and save predictions."
    )
    parser.add_argument("--input_path", required=True, help="Path to prepared input NetCDF")
    parser.add_argument("--output_path", required=True, help="Path to write predictions NetCDF")
    parser.add_argument("--model_name", default="v1/deterministic_2_8_deg.pkl",
                        help=("Checkpoint under gs://neuralgcm/models/ "
                              "(e.g. v1/deterministic_0_7_deg.pkl, v1/deterministic_1_4_deg.pkl, "
                              "v1/deterministic_2_8_deg.pkl, v1/stochastic_1_4_deg.pkl, "
                              "v1_precip/stochastic_precip_2_8_deg.pkl, v1_precip/stochastic_evap_2_8_deg.pkl)"))
    parser.add_argument("--inner_steps", type=int, default=6,
                        help="Hours between saved outputs (e.g., 6 -> every 6h)")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Total number of saved outputs (if not set, defaults to 4*6//inner_steps)")

    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    # Load model and input dataset
    model = load_model(args.model_name)
    eerie = xr.open_dataset(args.input_path)

    # Regrid to model grid (same logic as your snippet)
    regridder = build_regridder(eerie, model.data_coords.horizontal)
    eval_eerie = xarray_utils.regrid(eerie, regridder)
    eval_eerie = xarray_utils.fill_nan_with_nearest(eval_eerie)

    # Time stepping logic (as in the snippet)
    inner_steps = args.inner_steps
    if args.num_steps is None:
        outer_steps = 4 * 6 // inner_steps  # same default logic as provided snippet
    else:
        outer_steps = int(args.num_steps)

    timedelta = np.timedelta64(1, "h") * inner_steps
    # lead times in hours since initialization
    lead_hours = (np.arange(outer_steps, dtype=np.int64) * inner_steps)

    # Initialize model state
    inputs = model.inputs_from_xarray(eval_eerie.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_eerie.isel(time=1))
    rng_key = jax.random.key(args.seed)
    initial_state = model.encode(inputs, input_forcings, rng_key)

    # Use persistence for forcing variables (SST & sea ice cover)
    all_forcings = model.forcings_from_xarray(eval_eerie.head(time=1))

    # Forecast
    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )

    # Convert model outputs to xarray with a numeric lead axis (hours)
    predictions_ds = model.data_to_xarray(predictions, times=lead_hours)

    # Build init_time from the ORIGINAL input dataset (not the regridded one)
    init0 = eerie.time.isel(time=0).values
    init_time_value = np.datetime64(init0, "ns")  # force datetime64[ns]

    # valid_time = init_time + lead_hours[h]
    valid_time_vals = (init_time_value + lead_hours.astype("timedelta64[h]")).astype("datetime64[ns]")

    # Assign coordinates: replace 'time' with valid_time; keep forecast_hour auxiliary; add init_time (scalar)
    predictions_ds = predictions_ds.assign_coords(
        time=("time", valid_time_vals),
        valid_time=("time", valid_time_vals),
        #forecast_hour=("time", lead_hours),
        init_time=init_time_value,
    )
    predictions_ds["time"].attrs.update({"standard_name": "time"})
    #predictions_ds["forecast_hour"].attrs.update({"long_name": "forecast lead time", "units": "hours since init_time"})

    # Save
    predictions_ds.to_netcdf(args.output_path)


if __name__ == "__main__":
    main()