import numpy as np
from numba import njit, i8, f8
from utils import argtopk


@njit(
    f8[::1](i8, i8[::1])
)
def do_planning(  # --> define {+0,+1} sustainable promotion utility
        J,
        croi,
):
    advt_preferences = np.ones(J, dtype="f8")
    advt_preferences[croi] = 0
    return advt_preferences


@njit(
    i8[::1](f8[::1], f8[::1], i8, f8)
)
def recommend_with_void_search_decoder(
        # ----------------------------< ubm
        user_preferences,             # user utility (from underlying RS) for each POI in the universe (size |pois|)
        advt_preferences,             # advt utility
        # ----------------------------< 3rd
        pack_size,                    # proposition (recommendation bundle) size
        lamb,                         # my optimisation trade off
):
    assert (0 <= lamb <= 1), "Lambda is out of range!"

    # estimate gumbel standard deviation
    _RERANKING_BETA = 0.001 * (6**0.5 / np.pi)

    # estimate gumbel noise
    _r = _RERANKING_BETA * (-1) * np.log(-np.log( np.random.random(size=len(user_preferences)) ))
    _s = _RERANKING_BETA * (-1) * np.log(-np.log( np.random.random(size=len(user_preferences)) ))

    # multistakeholder recommendation with stochastic re-ranking to break utility ties
    pack = argtopk((1-lamb) * (user_preferences+_r) + lamb * (advt_preferences+_s), k=pack_size)

    return pack
