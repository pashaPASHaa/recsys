#!/bin/bash
city="Istanbul"
OMP_NUM_THREADS=4 TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4 \
python3 -u ./src/exp/run2.py \
        --util_artefacts_file out/${city}_util.hdf5 \
        --aset_artefacts_file out/${city}_aset.hdf5 \
        --seed 2024 \
        2>&1 | tee log/${city}_run2_training.log
