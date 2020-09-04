#!/usr/bin/env bash

python run.py --config=configs/baseline.json
python run.py --config=configs/pooled.json
python run.py --config=configs/width.json
python run.py --config=configs/pooled_width.json