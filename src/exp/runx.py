import argparse
import copy
import os
import pickle
import numpy as np

from datafactory import load_lbsn_artefacts
from sim import simulate_1_day
from utils import haversine, seedutils


def run_experiment(  # --> high level configuration for `simulate_1_day` experiment simulation
        lbsn_file: str,
        util_file: str,
        aset_file: str,
        rate: float,
        args: dict[str,any],  # params to variate in experiment
):

    args = copy.deepcopy(args)
    print(f"\nStarting experiment with params: "
          f"key={args['key']} "
          f"aset_key={args['aset_key']} "
          f"recommender_system_key={args['recommender_system_key']} "
          f"salience_boost={args['salience_boost']:.2f} "
          f"rate={rate:.2f} "
          f"\n")

    experiment: dict[str,dict|list] = {
        "args": args,
        "improvement_user": [],  # more is better
        "improvement_harm": [],  # more is better
        "bias_improvement_user": [],  # 0 is better
        "bias_improvement_harm": [],  # 0 is better
        "hist": []}

    for lamb in args["lamb_arr"]:
        # run simulation with booststrapped users
        hist = simulate_1_day(
            lbsn_file,
            util_file,
            aset_file,
            rate=rate,
            args={
                "key": args["key"],
                "aset_key": args["aset_key"],
                "recommender_system_key": args["recommender_system_key"],
                "lamb": lamb,
                "pack_size": 8,
                "salience_boost": args["salience_boost"],
                "seed": args["seed"],
                "harm_collector_step": 1000,
                "disp_collector_step": 1000,
                "croi": args["croi"],
            }
        )
        # per user uplift
        void_user = hist["void_user"]
        nors_user = hist["nors_user"]
        data_user = hist["data_user"]
        improvement_user = np.mean((void_user - nors_user) / nors_user)

        # environmental cumulative sustainable uplift
        void_harm = hist["void_harm"][-1]
        nors_harm = hist["nors_harm"][-1]
        data_harm = hist["data_harm"][-1]
        improvement_harm = (-1) * ((void_harm - nors_harm) / nors_harm)

        # simulation bias
        bias_improvement_user = np.mean((nors_user - data_user) / data_user)
        bias_improvement_harm = (-1) * ((nors_harm - data_harm) / data_harm)

        # update
        experiment["improvement_user"].append(improvement_user)
        experiment["improvement_harm"].append(improvement_harm)
        experiment["bias_improvement_user"].append(bias_improvement_user)
        experiment["bias_improvement_harm"].append(bias_improvement_harm)
        experiment["hist"].append({
            __k: hist[__k] for __k in ("void_choice_arr",
                                       "nors_choice_arr",
                                       "data_choice_arr", "void_exposure_arr")
        })

        print(f"---------------------------------\n"
              f"lamb={lamb:.4f} | "
              f"user lift={improvement_user:.4f} | "
              f"sust lift={improvement_harm:.4f} | "
              f"user bias={bias_improvement_user:.4f} and sust bias={bias_improvement_harm:.4f} > "
              f"\n\n")

    return experiment


if __name__ == "__main__":

    # -------------------------------------------------------------------------

    # python3 runx.py --help
    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument("--lbsn_artefacts_file", type=str, required=True, help="file with LBSN trajectories")
    parser.add_argument("--util_artefacts_file", type=str, required=True, help="file with estimated user preferences")
    parser.add_argument("--aset_artefacts_file", type=str, required=True, help="file with ground truth user data")
    parser.add_argument("--dump_file", type=str, required=True, help="file to save dump")
    parser.add_argument("--key", type=str, required=True, help="recommendation algorithm to consider as ground truth")
    parser.add_argument("--recommender_system_key", type=str, required=True, help="recommendation algorithm")
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    seedutils(args.seed)

    # -------------------------------------------------------------------------

    # load dataset with lsbn trajectories and environment structure
    y, _, _, envstruct = load_lbsn_artefacts(args.lbsn_artefacts_file)
    city_centre = envstruct["city_centre"]
    city_radius = envstruct["city_radius"]
    size = y.shape[1]
    croi = []
    for j in range(size):
        if (haversine(city_centre, envstruct["location"][j,:]) < 0.2*city_radius) or (j < 0.2*size):
            croi += [j]
    croi = np.array(croi, dtype="i8")

    print(f"Detected script configuration for experiment:\n"
          f"key={args.key} "
          f"recommender_system_key={args.recommender_system_key} "
          f"seed={args.seed} "
          f"number of croi POIs={len(croi)} (with {len(croi)/size:.2f} from total universe)")
    print(croi)

    # -------------------------------------------------------------------------

    lamb_arr = np.linspace(0.0, 1.0, 41)  # uniform DOM[0:1]

    # -------------------------------------------------------------------------

    experiment_arr = {}
    for k_gathered in (2, 3, 4):
        experiment_arr[k_gathered] = {}
        for aset_key in (4, 8, 12, 16, 20):
            experiment = run_experiment(
                args.lbsn_artefacts_file,
                args.util_artefacts_file,
                args.aset_artefacts_file,
                rate=13,
                args={"key": f"{args.key}",
                      "aset_key": f"{aset_key}",
                      "recommender_system_key": (f"{args.recommender_system_key}", f"{k_gathered}"),
                      "croi": croi,
                      "seed": args.seed,
                      "salience_boost": 0.0,
                      "lamb_arr": lamb_arr}
            )
            experiment_arr[k_gathered][aset_key] = experiment

    # -------------------------------------------------------------------------

    # create leaf directory if it does not exist
    os.makedirs(os.path.dirname(args.dump_file), exist_ok=True)

    with open(args.dump_file, "wb") as f:
        pickle.dump(experiment_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------------------------------------------------

    print(f"DONE!")
