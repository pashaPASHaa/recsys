from numpy import log, union1d
from numba import njit, i8, f8
from utils import argtopk, mysigmoid, mysoftmax, rand_choice_arr_nb


# choice model: rational (aka. argtopk) user behaviour

@njit(
    i8[::1](f8[::1], i8[::1], i8[::1], i8)
)
def make_rational_choices(
        user_preferences,
        pack,
        aset,
        k,
):
    # extend awareness set with provided recommendations
    aset = union1d(aset, pack)
    # select
    c = aset[argtopk(user_preferences[aset], k=k)]
    return c


# choice model: repeated multinomial choice from awareness set without repetitions

@njit(
    i8[::1](f8[::1], i8[::1], i8[::1], i8, f8)
)
def make_boltzman_choices(
        user_preferences,
        pack,
        aset,
        k,
        beta,
):
    # extend awareness set with provided recommendations
    aset = union1d(aset, pack)
    # select
    c = aset[rand_choice_arr_nb(p=mysoftmax(beta*user_preferences[aset]), size=k)]
    return c
