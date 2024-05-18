import collections
import numba as nb
import numpy as np

from numba import njit, i8, f8


# environment constants
# -----------------------------------------------------------------------------


TIME_MORNING =  0*60  # time [min] from the beginning of the day
TIME_EVENING = 18*60  # time [min] from the beginning of the day
TIME_24HOURS = 24*60  # time [min] in the day


np.random.seed(1)  # reproducibility


# environment constants and structure
# -----------------------------------------------------------------------------


Obj = collections.namedtuple(
    "Obj",
    "location, capacity",
)

Poi = collections.namedtuple(
    "Poi",
    "location, capacity, time_required",
)

nb_Loc = i8[::1]  # array with [x, y]
nb_Obj = nb.types.NamedTuple([nb_Loc, i8], Obj)
nb_Poi = nb.types.NamedTuple([nb_Loc, i8, i8], Poi)


# user arrival dynamics model
# -----------------------------------------------------------------------------


def users_arrival_generator(rate: float, seed: int):  # rate [n/min]
    """
    Yields (t, f) usage logic:
        0 - continue
        1 - generate user class and provide itinerary recommendation
    """

    # init generator
    rng = np.random.default_rng(seed)

    # discretisation time step [min] for simulator
    h = 1/60

    t = 0
    while t < TIME_MORNING:
        yield (t, 0)
        t += h

    while t < TIME_EVENING:

        # time when user will arrive
        tta = t - 1/rate * np.log(1-rng.uniform())

        # deny late entry time
        if tta >= TIME_EVENING:
            break

        # idle till time when user will arrive
        while t < tta:
            yield (t, 0)
            t += h
        else:
            yield (t, 1)
            t += h

    while t < TIME_24HOURS:
        yield (t, 0)
        t += h
