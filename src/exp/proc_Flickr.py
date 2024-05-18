import numpy as np
import polars as pl

from datafactory import dump_lbsn_artefacts


PART_MIN_DISTRIBUTED_ACTIVITY = 1.25  # average number of visited items per part of a tour
PART_MAX_NUMS = 14                    # max number of part chunks a tourist can complete (otherwise a local)
TOUR_MIN_USER_ACTIVITY = 5            # min number of visited items per tour
TOUR_MAX_USER_ACTIVITY = 50           # max number of visited items per tour
ITEM_MIN_FEEDBACK = 10                # control for niche items


if __name__ == "__main__":

    city_arr = ("Rome", "Florence", "Pisa")

    city_radius_km_dict = {  # only historical city radius
        "Rome": 5,
        "Florence": 2,
        "Pisa": 2,
    }

    for city in city_arr:

        # ---------------------------------------------------------------------

        data: dict[str|int,any] = {}
        with open(f"data/tripbuilder-dataset-dist/{city.lower()}/{city.lower()}-trajectories.txt", "r") as file:
            for line in file:

                # --- proc one line

                part = {
                    "user_key": None,
                    "item_arr": [],
                    "time_arr": [],
                }
                for i, chunk in enumerate(line.rstrip("\n").split("\t")):
                    # parse user
                    if i == 0:
                        part["user_key"] = int(chunk)
                    # parse user visited trajectory within tau hours interval
                    else:
                        item_id, _, t, _ = [int(s) for s in chunk.split(";")]
                        part["item_arr"].append(item_id)
                        part["time_arr"].append(t)

                # --- update / merge with the current dataset

                curr_user_id = part.pop("user_key")
                if curr_user_id in data:
                    assert part["time_arr"][0] > data[curr_user_id][-1]["time_arr"][-1], "Bad time partition"
                    data[curr_user_id].append(part)
                else:
                    data[curr_user_id] = []
                    data[curr_user_id].append(part)

        # ---------------------------------------------------------------------

        tourist_data = {}
        glob_user_id = 0
        good_tourist_sessions = 0

        bad = 0
        for user_id in data:

            # --- detect local resident vs. tourist activity?

            s = 0
            n = 0
            init_time_arr = []
            for part in data[user_id]:
                part_item_arr = part["item_arr"]
                part_time_arr = part["time_arr"]
                s += len(part_item_arr)
                n += 1
                init_time_arr += [min(part_time_arr)]

            # --- filter

            if (s/n < PART_MIN_DISTRIBUTED_ACTIVITY) or (n > PART_MAX_NUMS):
                bad += 1
                continue

            # --- update tourist dataset

            tourist_data[glob_user_id] = {
                "item_arr": [],
                "time_arr": []}

            for part in data[user_id]:  # --> iterate over all parts for a particular user
                part_item_arr = part["item_arr"]
                part_time_arr = part["time_arr"]
                tourist_data[glob_user_id]["item_arr"].extend(part_item_arr)
                tourist_data[glob_user_id]["time_arr"].extend(part_time_arr)
                good_tourist_sessions += 1
            glob_user_id += 1

        print(f"Processed {len(data)} users, discarded {bad} locals, "
              f"independent tourist visits {glob_user_id}, sessions {good_tourist_sessions}")

        # ---------------------------------------------------------------------

        # tourist_data = {glob_user_id: {"item_arr": [],
        #                                "time_arr": []}
        #                }

        item_to_rank = {}
        rank_to_item = {}
        rank = 0
        for user in tourist_data:
            for item in tourist_data[user]["item_arr"]:
                if item in item_to_rank:
                    pass
                else:
                    item_to_rank[item] = rank
                    rank_to_item[rank] = item
                    rank += 1

        N = len(tourist_data)
        J = len(rank_to_item)

        # --- populate choices and item coordinates

        y = np.zeros((N, J), dtype="i8")
        n = 0
        for user in tourist_data:
            for item in tourist_data[user]["item_arr"]:
                y[n,item_to_rank[item]] = 1
            n += 1

        # ---

        tb_item = \
            pl.read_csv(f"data/tripbuilder-dataset-dist/{city.lower()}/{city.lower()}-pois-clusters.txt")

        location = np.zeros((J, 2), dtype="f8")
        for rank in range(J):
            r = tb_item.filter(pl.col("clus") == rank_to_item[rank])
            location[rank,0] = r["latitude"].item()
            location[rank,1] = r["longitude"].item()

        city_centre = np.mean(location, axis=0)
        city_radius = city_radius_km_dict[city]

        # --- do 1st filter (outliers)

        for _ in range(5):
            good_user_ma = (TOUR_MIN_USER_ACTIVITY <= y.sum(1)) & (y.sum(1) <= TOUR_MAX_USER_ACTIVITY)
            good_item_ma = (ITEM_MIN_FEEDBACK <= y.sum(0))
            y = y[good_user_ma][:,good_item_ma]
            location = location[good_item_ma,:]

        # --- do 2nd filter (popularity item reranking)

        pop_ranking_ix = np.argsort(-y.sum(axis=0))
        y = y[:,pop_ranking_ix]
        location = location[pop_ranking_ix,:]

        # --- save

        N, J = y.shape
        f = np.eye(J, dtype="f8")
        fnames = np.array([f"f{j:>03}" for j in range(J)])

        dump_lbsn_artefacts(
            file=f"out/{city}_lbsn.hdf5",
            y=y,
            f=f,
            fnames=fnames.astype("S8"),
            envstruct={
                "city_centre": city_centre,
                "city_radius": city_radius,
                "location": location,
                "capacity": np.array([1]*J, dtype="i8"),
                "time_required": np.array([1]*J, dtype="i8")
            },
        )
        print(f"Done! city={city} N={N} J={J}")
