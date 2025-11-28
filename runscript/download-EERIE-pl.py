#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
from functions import parse_arguments, read_config
import os

def main():
    args = parse_arguments()
    config = read_config(args.config)

    start_date = config["start_time"]
    output_path = config.get("output_path", ".")
    os.makedirs(output_path, exist_ok=True)

    target_pl = os.path.join(output_path, f"output_{start_date}_pl.grib")

    server = ECMWFDataServer()
    server.retrieve({
        "activity": "cmip6",
        "class": "ed",
        "dataset": "research",
        "date": f"{start_date}",
        "experiment": "hist",
        "expver": "0002",
        "generation": "1",
        "grid": "1/1",
        "levelist": "1/5/10/20/30/50/70/100/150/200/250/300/400/500/600/700/850/925/1000",
        "levtype": "pl",
        "model": "ifs",
        "param": "129/130/131/132/133/157", # Geopotential/Temperature/U component of wind/V component of wind/Specific humidity
        "realization": "1",
        "resolution": "high",
        "stream": "clte",
        "target": target_pl,
        "time": "00:00:00/06:00:00",
        "type": "fc"
    })

if __name__ == "__main__":
    main()