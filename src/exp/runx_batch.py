import argparse
import os
import subprocess


SEED = 2024


def make_runx_instruction(this_path, city, key, recommender_system_key):
    runner = (
        f"python3 {this_path}/runx.py "
        f"--lbsn_artefacts_file {this_path}/../../out/{city}_lbsn.hdf5 "
        f"--util_artefacts_file {this_path}/../../out/{city}_util.hdf5 "
        f"--aset_artefacts_file {this_path}/../../out/{city}_aset.hdf5 "
        f"--dump_file {this_path}/../../out/experiments/runx_{city}_{key}_{recommender_system_key}.pk "
        f"--key {key} "
        f"--recommender_system_key {recommender_system_key} "
        f"--seed {SEED} "
        f"2>&1 | tee {this_path}/../../log/runx_{city}_{key}_{recommender_system_key}.log"
    )
    return runner


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch Experiment")
    parser.add_argument("--city", type=str, required=True, help="City of interest: Rome, Florence, Istanbul")
    args = parser.parse_args()

    this_path = os.path.dirname(__file__)
    keys = ["pop", "vae", "wmf", "bpr", "ngcf", "ease"]  # design of experiment

    processes = []
    for key in keys:
        for recommender_system_key in keys:
            print(f"Starting experiment in city={args.city} with configuration parameters "
                  f"key={key:>4} "
                  f"recommender_system_key={recommender_system_key:>4}")
            proc = subprocess.Popen(make_runx_instruction(
                this_path=this_path, city=args.city, key=key, recommender_system_key=recommender_system_key),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True)
            processes.append(proc)
    [proc.wait() for proc in processes]
    print(f"DONE!")
