#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--num",
        type=int,
        default=1000
    )

    args = parser.parse_args()


    with open(args.input, "r") as f:
        data=json.load(f)


    if len(data) < args.num:
        raise ValueError(
            f"Dataset only has {len(data)} samples"
        )


    subset=data[:args.num]


    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(args.output,"w") as f:

        json.dump(
            subset,
            f,
            indent=2
        )


    print("================")
    print("source:",args.input)
    print("output:",args.output)
    print("total:",len(data))
    print("subset:",len(subset))
    print("================")



if __name__=="__main__":
    main()