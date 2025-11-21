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


def regrid_and_fill(src: xr.Dataset, regridder) -> xr.Dataset:
    out = xarray_utils.regrid(src, regridder)
    out = xarray_utils.fill_nan_with_nearest(out)
    return out


def run_forecast(model,
                 eval_eerie: xr.Dataset,
                 eerie_input: xr.Dataset,
                 inner_steps: int,
                 forecast_days: int,
                 seed: int = 42) -> xr.Dataset:
    """
    Run the model forecast and build output dataset with correct init_time and valid_time.
    """
    outer_steps = (forecast_days * 24) // inner_steps
    timedelta = np.timedelta64(1, "h") * inner_steps
    lead_hours = np.arange(outer_steps, dtype=np.int64) * inner_steps  # forecast lead hours

    # --- Model initialization ---
    inputs = model.inputs_from_xarray(eval_eerie.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_eerie.isel(time=1))
    rng_key = jax.random.key(seed)
    initial_state = model.encode(inputs, input_forcings, rng_key)

    all_forcings = model.forcings_from_xarray(eval_eerie.head(time=1))

    # --- Run model ---
    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )

    predictions_ds = model.data_to_xarray(predictions, times=lead_hours)

    # --- Build init_time and valid_time from original eerie dataset ---
    init0 = eerie_input.time.isel(time=0).values
    init_time_value = np.datetime64(init0, 'ns')  # ensure datetime64[ns]

    lead_td = lead_hours.astype('timedelta64[h]')
    valid_time_vals = (init_time_value + lead_td).astype('datetime64[ns]')

    init_time = xr.DataArray(
        init_time_value,
        dims=(),
        attrs={"long_name": "model initialization time"}
    )
    valid_time = xr.DataArray(
        valid_time_vals,
        dims=("time",),
        attrs={"long_name": "valid time"}
    )
    forecast_hour = xr.DataArray(
        lead_hours,
        dims=("time",),
        attrs={"long_name": "forecast lead time", "units": "hours since init_time"}
    )

    # --- Assign datetime coordinates ---
    predictions_ds = predictions_ds.assign_coords(
        time=("time", valid_time.values),
        valid_time=("time", valid_time.values),
        init_time=init_time,
        forecast_hour=forecast_hour,
    )
    predictions_ds["time"].attrs.update({"standard_name": "time"})

    return predictions_ds


def main():
    parser = argparse.ArgumentParser(
        description="Run NeuralGCM inference from prepared input layer and save predictions."
    )
    parser.add_argument("--input_path", required=True, help="Path to prepared input NetCDF")
    parser.add_argument("--output_path", required=True, help="Path to write predictions NetCDF")
    parser.add_argument("--model_name", default="v1/deterministic_2_8_deg.pkl",
                        help=("Model checkpoint name on GCS under gs://neuralgcm/models/. "
                              "Examples: v1/deterministic_0_7_deg.pkl, "
                              "v1/deterministic_1_4_deg.pkl, v1/deterministic_2_8_deg.pkl, "
                              "v1/stochastic_1_4_deg.pkl, v1_precip/stochastic_precip_2_8_deg.pkl, "
                              "v1_precip/stochastic_evap_2_8_deg.pkl"))
    parser.add_argument("--inner_steps", type=int, default=6,
                        help="Hours between saved outputs (e.g., 6 -> save every 6h)")
    parser.add_argument("--forecast_days", type=int, default=4,
                        help="Total forecast length in days")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    model = load_model(args.model_name)
    eerie = xr.open_dataset(args.input_path)

    regridder = build_regridder(eerie, model.data_coords.horizontal)
    eval_eerie = regrid_and_fill(eerie, regridder)

    predictions_ds = run_forecast(
        model,
        eval_eerie=eval_eerie,
        eerie_input=eerie,         # <--- pass original input dataset
        inner_steps=args.inner_steps,
        forecast_days=args.forecast_days,
        seed=args.seed,
    )

    predictions_ds.to_netcdf(args.output_path)


if __name__ == "__main__":
    main()