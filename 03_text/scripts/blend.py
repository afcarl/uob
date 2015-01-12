#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
import sys
import pandas as pd


def main(args):
    if len(args.inputs) <= 1:
        sys.stderr.write('Error: Invalid arguments\nMore than 1 input files required')
        return -1
    data = []
    for filename in args.inputs:
        name, ext = os.path.splitext(filename)
        if ext == '.gz':
            data.append(pd.read_csv(filename, compression='gzip', header=None))
        else:
            data.append(pd.read_csv(filename, header=None))
    for i in range(1, len(data)):
        data[0] += data[i]
    data[0] /= len(data)
    data[0].to_csv(args.output, float_format='%.10f', index=False, header=None)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    code = main(args)
    exit(code)
