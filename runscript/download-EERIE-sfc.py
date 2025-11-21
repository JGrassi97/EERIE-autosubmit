#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
from functions import parse_arguments, read_config
import os

def main():
    args = parse_arguments()
    config = read_config(args.config)

    start_date = config["start_time"]
    member = config["member"]
    output_path = config.get("output_path", ".")
    os.makedirs(output_path, exist_ok=True)

    target_sfc = os.path.join(output_path, f"output_{start_date}_r{member}_sfc.grib")

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
        "levtype": "sfc",
        "model": "ifs",
        "param": "31/34/78/79",
        "realization": f"{member}",
        "resolution": "high",
        "stream": "clte",
        "target": target_sfc,
        "time": "00:00:00/06:00:00",
        "type": "fc"
    })

if __name__ == "__main__":
    main()