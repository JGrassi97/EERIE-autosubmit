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
        description="Run NeuralGCM inference (stochastic ensemble) and save predictions."
    )
    parser.add_argument("--input_path", required=True, help="Path to prepared input NetCDF")
    parser.add_argument("--output_path", required=True, help="Path to write predictions NetCDF")
    parser.add_argument("--model_name", default="v1/stochastic_1_4_deg.pkl",
                        help=("Checkpoint under gs://neuralgcm/models/. "
                              "Use a stochastic model for ensemble (e.g. v1/stochastic_1_4_deg.pkl, "
                              "v1_precip/stochastic_precip_2_8_deg.pkl, v1_precip/stochastic_evap_2_8_deg.pkl)"))
    parser.add_argument("--inner_steps", type=int, default=6,
                        help="Hours between saved outputs (e.g., 6 -> every 6h)")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Total number of saved outputs; default = 10*6//inner_steps (like your snippet)")
    parser.add_argument("--n_members", type=int, default=3,
                        help="Number of ensemble members")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; different members use different splits of this key")

    args = parser.parse_args()

    # Load model and input dataset
    model = load_model(args.model_name)
    eerie = xr.open_dataset(args.input_path)

    # Regrid to model grid
    regridder = build_regridder(eerie, model.data_coords.horizontal)
    eval_eerie = xarray_utils.regrid(eerie, regridder)
    eval_eerie = xarray_utils.fill_nan_with_nearest(eval_eerie)

    # Time stepping (match your stochastic example: default 10 steps at inner_steps cadence)
    inner_steps = args.inner_steps
    if args.num_steps is None:
        outer_steps = 10 * 6 // inner_steps
    else:
        outer_steps = int(args.num_steps)

    timedelta = np.timedelta64(1, "h") * inner_steps
    lead_hours = np.arange(outer_steps, dtype=np.int64) * inner_steps  # [0, 6, 12, ...]

    # Build datetimes from ORIGINAL input (not regridded)
    init0 = eerie.time.isel(time=0).values
    init_time_value = np.datetime64(init0, "ns")
    valid_time_vals = (init_time_value + lead_hours.astype("timedelta64[h]")).astype("datetime64[ns]")

    # Common inputs/forcings (state init differs by RNG)
    inputs = model.inputs_from_xarray(eval_eerie.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_eerie.isel(time=1))
    all_forcings = model.forcings_from_xarray(eval_eerie.head(time=1))

    # Build ensemble keys
    base_key = jax.random.key(args.seed)
    keys = jax.random.split(base_key, num=args.n_members)

    members = []
    member_ids = np.arange(args.n_members, dtype=np.int32)

    for i, key in enumerate(keys):
        # Encode initial state with member-specific RNG key
        initial_state = model.encode(inputs, input_forcings, key)

        # Forecast
        final_state, predictions = model.unroll(
            initial_state,
            all_forcings,
            steps=outer_steps,
            timedelta=timedelta,
            start_with_input=True,
        )

        # Convert to xarray using VALID datetimes as the time axis
        ds_m = model.data_to_xarray(predictions, times=valid_time_vals).compute()

        # Attach time coords: init_time (scalar datetime), valid_time (== time), forecast_hour (int hours)
        ds_m = ds_m.assign_coords(
            init_time=init_time_value,
            valid_time=("time", valid_time_vals),
            forecast_hour=("time", lead_hours),
        )
        ds_m["time"].attrs.update({"standard_name": "time"})
        ds_m["forecast_hour"].attrs.update({"long_name": "forecast lead time", "units": "hours since init_time"})

        # Expand with ensemble_member dimension and add seed as coord for traceability
        ds_m = ds_m.expand_dims(ensemble_member=[member_ids[i]])
        ds_m = ds_m.assign_coords(seed=("ensemble_member", np.array([args.seed + i], dtype=np.int64)))

        members.append(ds_m)

    # Concatenate all members
    predictions_ds = xr.concat(members, dim="ensemble_member")
    predictions_ds = predictions_ds.assign_coords(
        ensemble_member=("ensemble_member", member_ids)
    )

    # Save
    predictions_ds.to_netcdf(args.output_path)


if __name__ == "__main__":
    main()