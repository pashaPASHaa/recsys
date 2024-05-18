#!/bin/bash
city="Florence"
OMP_NUM_THREADS=4 TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=4 \
python3 -u ./src/exp/run1.py \
        --lbsn_artefacts_file out/${city}_lbsn.hdf5 \
        --util_artefacts_file out/${city}_util.hdf5 \
        2>&1 | tee log/${city}_run1_training.log
