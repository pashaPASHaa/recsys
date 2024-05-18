import numpy as np
import polars as pl

from numba import njit, i8, f8
from datafactory import dump_lbsn_artefacts
from utils import haversine


PART_TAU = 8                          # max time (hours) threshold to build part of a tour
PART_MIN_DISTRIBUTED_ACTIVITY = 1.25  # average number of visited items per part of a tour
PART_MAX_NUMS = 14                    # max number of part chunks a tourist can complete (otherwise a local)
TOUR_MIN_USER_ACTIVITY = 5            # min number of visited items per tour
TOUR_MAX_USER_ACTIVITY = 50           # max number of visited items per tour
ITEM_MIN_FEEDBACK = 10                # control for niche items


@njit(
    (f8[::1], f8[:,::1])
)
def haversine_batch(orig, dests):  # --> batch distance in km
    N = len(dests)
    dist_arr = np.empty(N, dtype="f8")
    for n in range(N):
        dist_arr[n] = haversine(orig, dests[n,:])
    return dist_arr


def read_user_tb(file):
    tb_user = (
        pl.scan_csv(
            source=file,
            has_header=False,
            new_columns=["user_id", "item_id", "time", "UTC_offset"],
            schema={
                "user_id": pl.String,
                "item_id": pl.String,
                "time": pl.String,
                "UTC_offset": pl.Int32,
            },
            separator="\t"
        )
        .select("user_id", "item_id", "time")
        .with_columns(pl.col("time").str.to_datetime("%a %b %d %H:%M:%S %z %Y", strict=False))
        .collect()
    )
    return tb_user


def read_item_tb(file):
    tb_item = pl.scan_csv(
        source=file,
        has_header=False,
        new_columns=["item_id", "lat", "lon", "category_name", "country_code"],
        schema={
            "item_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "category_name": pl.String,
            "country_code": pl.String,
        },
        separator="\t"
    ).collect()
    return tb_item


def read_city_tb(file):
    tb_city = pl.scan_csv(
        source=file,
        has_header=False,
        new_columns=["city", "lat", "lon", "country_code", "country_name", "city_type"],
        schema={
            "city": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "country_code": pl.String,
            "country_name": pl.String,
            "city_type": pl.String,
        },
        separator="\t"
    ).collect()
    return tb_city


if __name__ == "__main__":

    # --- city of interest (city_name, coutry_code, city_radius_km)

    city_arr = [
        ("Barcelona", "ES", 8),
        ("Copenhagen", "DK", 5),
        ("Istanbul", "TR", 12),
        ("London", "GB", 12),
        ("Lisbon", "PT", 5),
        ("Paris", "FR", 6),
    ]

    # --- load data

    tb_user = read_user_tb("data/Foursquare33M/dataset_TIST2015_Checkins.txt")
    tb_item = read_item_tb("data/Foursquare33M/dataset_TIST2015_POIs.txt")
    tb_city = read_city_tb("data/Foursquare33M/dataset_TIST2015_Cities.txt")

    print(tb_user.sample(4))
    print(tb_item.sample(4))
    print(tb_city.sample(4))

    good = [
        "Antique Shop", "Aquarium", "Art Gallery", "Art Museum", "Arts & Crafts Store", "Arts & Entertainment",
        "Beach", "Bookstore", "Bridge", "Building",
        "Campground", "Capitol Building", "Castle", "Cemetery", "Church", "Concert Hall", "Courthouse",
        "Fair", "Farm", "Farmers Market", "Flea Market",
        "Garden", "Garden Center", "General Entertainment", "General Travel",
        "Hiking Trail", "Historic Site", "History Museum", "Hot Spring",
        "Indie Movie Theater", "Indie Theater",
        "Jazz Club",
        "Lake", "Library",
        "Monument / Landmark", "Mosque", "Museum", "Music Venue", "Movie Theater", "Mountain",
        "Opera House", "Other Great Outdoors", "Outdoors & Recreation",
        "Park", "Performing Arts Venue", "Public Art",
        "River",
        "Scenic Lookout", "Science Museum", "Sculpture Garden", "Shrine", "Stables", "Stadium", "Synagogue",
        "Temple", "Theater", "Theme Park", "Travel Lounge", "Tourist Information Center",
        "Volcano", "Volcanoes",
        "Water Park",
        "Zoo",
    ]

    # --- proc city arr

    for city, country_code, radius_km in city_arr:

        # ---------------------------------------------------------------------

        orig = (  # --> get (lat, lon) for `city`
            tb_city
            .filter(pl.col("city") == city)
            .select(pl.concat_list("lat", "lon"))
            .item()
            .to_numpy()
            .copy()
        )

        # ---------------------------------------------------------------------

        def fn(orig, struct):
            dests = np.column_stack([
                struct.struct["lat"].to_numpy(),
                struct.struct["lon"].to_numpy(),
            ])
            return haversine_batch(orig, dests)

        # ---------------------------------------------------------------------

        tb = (  # --> item list in `city` of interest
            tb_item
            .filter(pl.col("country_code") == country_code)
            .with_columns(
                pl.struct("lat", "lon")
                .map_batches(lambda struct: fn(orig, struct) < radius_km).alias("in_city")
            )
            .filter(
                pl.col("in_city") &
                pl.col("category_name").is_in(good)
            )
            .get_column("item_id")
        )

        tb = (  # --> make user trajectories
            tb_user
            .filter(pl.col("item_id").is_in(tb))  # only the `city` of interest
            .drop_nulls()
            .sort("time")
            .group_by("user_id")
            .agg(
                pl.col("item_id").alias("item_id_arr"),
                pl.col("time").alias("time_arr"),
                pl.col("time").diff().alias("time_diff_arr")
            )
        )

        # ---------------------------------------------------------------------

        tourist_data = {}
        glob_user_id = 0
        good_tourist_sessions = 0

        bad = 0
        for step, row in enumerate(tb.iter_rows(named=True)):

            # --- chunks (aka parts) processor

            List = [
                {"part_item_arr": [],
                 "part_time_arr": []}
            ]
            for item_id, t, dt in zip(row["item_id_arr"], row["time_arr"], row["time_diff_arr"]):
                if dt is None:
                    tau = 0
                else:
                    tau = dt.total_seconds()//60//60  # --> hour
                assert (tau >= 0), "Detected trajectory permutation in data"
                if (tau < PART_TAU):
                    pass
                else:
                    List.append(
                        {"part_item_arr": [],
                         "part_time_arr": []}
                    )
                List[-1]["part_item_arr"].append(item_id)
                List[-1]["part_time_arr"].append(t)

            # --- detect local resident vs. tourist activity?

            s = 0
            n = 0
            for part in List:
                a = len(part["part_item_arr"])
                b = len(part["part_time_arr"])
                assert (a == b), "User part error"
                s += a
                n += 1
            assert s == len(row["time_arr"]), "User sequence of parts error"

            # --- filter

            if (s/n < PART_MIN_DISTRIBUTED_ACTIVITY) or (n > PART_MAX_NUMS):
                bad += 1
                continue

            # --- update tourist dataset

            tourist_data[glob_user_id] = {
                "tour_item_arr": [],
                "tour_time_arr": []}

            for part in List:  # --> iterate over all parts for a particular user
                part_item_arr = part["part_item_arr"]
                part_time_arr = part["part_time_arr"]
                tourist_data[glob_user_id]["tour_item_arr"].extend(part_item_arr)
                tourist_data[glob_user_id]["tour_time_arr"].extend(part_time_arr)
                good_tourist_sessions += 1
            glob_user_id += 1
        print(f"Processed {len(tb)} users, discarded {bad} locals, "
              f"independent tourist visits {glob_user_id}, sessions {good_tourist_sessions}")

        # ---------------------------------------------------------------------

        # tourist_data = {glob_user_id: {"tour_item_arr": [],
        #                                "tour_time_arr": []}
        #                }

        item_to_rank = {}
        rank_to_item = {}
        rank = 0
        for user in tourist_data:
            for item in tourist_data[user]["tour_item_arr"]:
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
            for item in tourist_data[user]["tour_item_arr"]:
                y[n,item_to_rank[item]] = 1
            n += 1

        location = np.zeros((J, 2), dtype="f8")
        for rank in range(J):
            r = tb_item.filter(pl.col("item_id") == rank_to_item[rank])
            location[rank,0] = r["lat"].item()
            location[rank,1] = r["lon"].item()

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
                "city_centre": orig,
                "city_radius": radius_km,
                "location": location,
                "capacity": np.array([1]*J, dtype="i8"),
                "time_required": np.array([1]*J, dtype="i8")
            },
        )
        print(f"Done! city={city} N={N} J={J}")
