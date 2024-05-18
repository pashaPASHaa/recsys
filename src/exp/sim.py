import time
import numpy as np

from datafactory import load_lbsn_artefacts, load_util_artefacts, load_aset_artefacts
from environment import users_arrival_generator
from recommender import do_planning, recommend_with_void_search_decoder
from ubm import make_boltzman_choices
from utils import argnonz


def simulate_1_day(
        lbsn_file: str,
        util_file: str,
        aset_file: str,
        rate: float,
        args: dict[str,any],
):
    # -------------------------------------------------------------------------

    # parameters that define:
    # - ground truth preference extraction mechanism
    # - size of awareness set
    # - algorithm to be used as a recommender engine
    key, aset_key = args["key"], args["aset_key"]
    recommender_system_key, recommender_system_t_key = args["recommender_system_key"]

    # parameters that may affect the simulation outcome
    lamb = args["lamb"]  # tradeoff (1-lamb) * user_u + lamb * sust_u
    pack_size = args["pack_size"]
    salience_boost = args["salience_boost"]
    seed = args["seed"]
    harm_collector_step = args["harm_collector_step"]  # how often collect harm
    disp_collector_step = args["disp_collector_step"]  # how often display info
    croi = args["croi"]

    # -------------------------------------------------------------------------

    # load dataset with user preferences
    true_data_map, pred_data_map = load_util_artefacts(util_file)

    # load dataset with user awareness set
    y_true, A_map = load_aset_artefacts(aset_file)

    assert np.all(true_data_map["y"] == y_true), f"Data mismatch between {util_file} and {aset_file} file"

    N, J = y_true.shape
    true_user_preferences = true_data_map[key]
    pred_user_preferences = pred_data_map[recommender_system_key][recommender_system_t_key]
    A, beta = A_map[key][aset_key]["A"], A_map[key][aset_key]["beta"]
    del true_data_map

    # -------------------------------------------------------------------------

    # sustainable promotion utility
    advt_preferences = do_planning(J, croi)

    # -------------------------------------------------------------------------

    # init structure
    void_harm = 0
    cond_harm = 0
    nors_harm = 0
    data_harm = 0

    log = {
        "void_user": [], "void_harm": [],
        "cond_user": [], "cond_harm": [],
        "nors_user": [], "nors_harm": [],
        "data_user": [], "data_harm": [],
        "n": [],
        "void_choice_arr": np.zeros(J, dtype="i8"), "void_exposure_arr": np.zeros(J, dtype="i8"),
        "cond_choice_arr": np.zeros(J, dtype="i8"), "cond_exposure_arr": np.zeros(J, dtype="i8"),
        "nors_choice_arr": np.zeros(J, dtype="i8"),
        "data_choice_arr": np.zeros(J, dtype="i8"),
    }

    # -------------------------------------------------------------------------

    for t, act in users_arrival_generator(rate=rate, seed=seed):

        tic = time.time()

        if not act:
            continue

        harm_ok = len(log["n"]) % harm_collector_step == 0
        disp_ok = len(log["n"]) % disp_collector_step == 0

        # ---------------------------------------------------------------------
        # SAMPLE USER FROM POPULATION WITH CORRESPONDING NUMBER OF ITEMS TO SEE

        n = np.random.randint(0, N)
        log["n"] += [n]
        need_size = y_true[n,:].sum()

        # ---------------------------------------------------------------------
        # RECOMMENDER SYSTEM FORESEES WHAT PACK TO RECOMMEND (AS A PROPOSITION)

        void_pack = recommend_with_void_search_decoder(pred_user_preferences[n,:], advt_preferences, pack_size, lamb)
        cond_pack = recommend_with_void_search_decoder(pred_user_preferences[n,:], advt_preferences, pack_size, lamb)
        nors_pack = np.array([], dtype="i8")
        # record recommendations
        log["void_exposure_arr"][void_pack] += 1
        log["cond_exposure_arr"][cond_pack] += 1

        # ---------------------------------------------------------------------
        # USER REACTS TO PROPOSED RECOMMENDATION PACK AND FOLLOWS THE ITINERARY

        anset = argnonz(A[n,:])

        boost = np.where(np.isin(np.arange(J), void_pack), salience_boost, 0.0)
        accepted_void_pack = make_boltzman_choices(true_user_preferences[n,:] + boost, void_pack, anset,
                                                   need_size, beta[n])

        boost = np.where(np.isin(np.arange(J), cond_pack), salience_boost, 0.0)
        accepted_cond_pack = make_boltzman_choices(true_user_preferences[n,:] + boost, cond_pack, anset,
                                                   need_size, beta[n])

        boost = 0
        accepted_nors_pack = make_boltzman_choices(true_user_preferences[n,:] + boost, nors_pack, anset,
                                                   need_size, beta[n])

        accepted_data_pack = argnonz(y_true[n,:])

        # -----------------------------

        # made choices increment
        log["void_choice_arr"][accepted_void_pack] += 1
        log["cond_choice_arr"][accepted_cond_pack] += 1
        log["nors_choice_arr"][accepted_nors_pack] += 1
        log["data_choice_arr"][accepted_data_pack] += 1

        # -----------------------------

        void_user = np.sum(true_user_preferences[n,accepted_void_pack])
        cond_user = np.sum(true_user_preferences[n,accepted_cond_pack])
        nors_user = np.sum(true_user_preferences[n,accepted_nors_pack])
        data_user = np.sum(true_user_preferences[n,accepted_data_pack])
        # record utilities
        log["void_user"].append(void_user)
        log["cond_user"].append(cond_user)
        log["nors_user"].append(nors_user)
        log["data_user"].append(data_user)

        # -----------------------------

        if harm_ok:
            void_harm = np.sum(log["void_choice_arr"][croi])
            cond_harm = np.sum(log["cond_choice_arr"][croi])
            nors_harm = np.sum(log["nors_choice_arr"][croi])
            data_harm = np.sum(log["data_choice_arr"][croi])
            # record utilities
            log["void_harm"].append(void_harm)
            log["cond_harm"].append(cond_harm)
            log["nors_harm"].append(nors_harm)
            log["data_harm"].append(data_harm)

        # -----------------------------

        toc = time.time()

        if disp_ok:
            print(
                f"{len(log['n']):>05d} | {n:>04d} | {toc-tic:>6.4f} sec  k={need_size:>2d}"
                f"  |vcn|  "
                f"util= "
                f"{void_user:>5.2f}  "
                f"{cond_user:>5.2f}  "
                f"{nors_user:>5.2f}  |  "
                f"harm= "
                f"{void_harm:>6.0f}  "
                f"{cond_harm:>6.0f}  "
                f"{nors_harm:>6.0f}  "
            )

    return {key: np.asarray(val) for key, val in log.items()}


if __name__ == "__main__":

    import os

    # -------------------------------->

    path = os.path.dirname(os.path.realpath(__file__))
    lbsn_file = os.path.join(path, f"../../out/Rome_lbsn.hdf5")
    util_file = os.path.join(path, f"../../out/Rome_util.hdf5")
    aset_file = os.path.join(path, f"../../out/Rome_aset.hdf5")

    # -------------------------------->

    croi = np.arange(25)  # first 25 POIs are considered as unsustainable

    # -------------------------------->

    simulate_1_day(
        lbsn_file,
        util_file,
        aset_file,
        rate=2.0,
        args={
            "key": "bpr",
            "aset_key": "10",
            "recommender_system_key": ("wmf", "4"),
            "lamb": 0.4,
            "pack_size": 8,
            "salience_boost": 0.01,
            "seed": 1,
            "harm_collector_step": 1,
            "disp_collector_step": 1,
            "croi": croi,
        }
    )
