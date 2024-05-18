import numpy as np
from numba import njit, b1, i8, f8


@njit
def seedutils(seed):
    np.random.seed(seed)


@njit(
    [i8[::1](i8[::1], i8), i8[::1](f8[::1], i8)]
)
def argtopk(a, k):
    # find unsorted top-`k` indices in array `a`
    n = len(a)
    assert (k >  0), "Bad k (too small)"
    assert (k <= n), "Bad k (too large)"
    o = np.argpartition(a, n-k)[n-k:]
    assert len(o) == k, "Temporary check!"
    return o


@njit(
    [i8[::1](b1[::1]), i8[::1](i8[::1]), i8[::1](f8[::1])]
)
def argnonz(a):
    o = np.flatnonzero(a)
    return o


@njit(
    f8[::1](f8[::1])
)
def mysigmoid(x):
    ex = 1 / (1+np.exp(-x))
    return ex


@njit(
    f8[::1](f8[::1])
)
def mysoftmax(x):
    ex  = np.exp(x-np.max(x))
    ex /= ex.sum()
    return ex  # safe exponent


@njit(
    i8[::1](f8[::1])
)
def data2rank(x):
    o = x.argsort()  # order
    r = o.argsort()  # ranks
    return r


@njit(
    f8(i8[::1], i8[::1])
)
def myprecision(
        c_true,
        c_pred,
):
    return len(set(c_true) & set(c_pred)) / len(set(c_pred))


@njit(
    f8(i8[::1], i8[::1])
)
def myrecall(
        c_true,
        c_pred,
):
    return len(set(c_true) & set(c_pred)) / len(set(c_true))


def mygini(a):
    assert np.all(a >= 0), "Only non-negative counts are allowed!"
    # normalise counts to probability
    a = a / np.sum(a)
    # mean absolute difference
    mad = np.abs(np.subtract.outer(a, a)).mean()
    # Gini coefficient
    g = 0.5 * mad / np.mean(a)
    return g


@njit(
    f8(i8[::1], i8[::1])
)
def myoverlap(A, B):
    n = len(A)
    assert n == len(B), "Shape mismatch"
    score = len(np.intersect1d(A, B)) / n
    return score


@njit(
    f8(i8[::1], i8[::1], f8)
)
def myrbo(A, B, p):  # --> calculate rank biased overlap similarity between two rank lists
    """
    [2010] A Similarity Measure for Indefinite Rankings, ACM Transactions on Information Systems
        William Webber and Alistair Moffat and Justin Zobel
    """
    n = len(A)
    assert n == len(B), "Shape mismatch"
    w = (1-p)/(1-p**n)
    s = 0
    a = set()
    b = set()
    score = 0
    for k in range(n):
        if A[k] == B[k]:
            h = 1
        else:
            h_ab = int(A[k] in b)
            h_ba = int(B[k] in a)
            h = h_ab + h_ba
        a.add(A[k])
        b.add(B[k])
        s += h
        score += s/(k+1) * w
        w *= p
    return score


@njit(
    b1(f8[::1])
)
def ok(p):
    f = (np.abs(1-np.sum(p)) < 1e-8)
    return f


# @njit(
#     i8(f8[::1])
# )
# def rand_choice_nb(p):
#     assert ok(p), "Bad input"
#     r = np.searchsorted(np.cumsum(p), np.random.random(), side="right")
#     return r


@njit(
    i8(f8[::1])
)
def rand_choice_nb(p):
    assert ok(p), "Bad input"
    x = np.random.random()
    c = p[0]
    i = 0
    while c < x:
        i += 1
        c += p[i]
    return i


@njit(
    i8[::1](f8[::1], i8)
)
def rand_choice_arr_nb(p, size):

    # http://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/#comment-4191330451

    assert ok(p), "Bad input"
    # random uniform
    u = np.random.random(len(p))
    # gumbel top-k
    r_arr = argtopk(np.log(p) - np.log(-np.log(u)), k=size)
    return r_arr


@njit(
    [i8(i8[::1]), f8(f8[::1])]
)
def rand_member_nb(a):
    n = len(a)
    r = a[rand_choice_nb(p=np.ones(n)/n)]
    return r


@njit(
    [i8[::1](i8[::1], i8), f8[::1](f8[::1], i8)]
)
def rand_member_arr_nb(a, size):
    n = len(a)
    r = a[rand_choice_arr_nb(p=np.ones(n)/n, size=size)]
    return r


@njit(
    (f8[::1], f8[::1])
)
def haversine(orig, dest):
    """
    orig: (lat, lon)
    dest: (lat, lon)
    Calculates the great-circle (as-the-crow-flies) distance between two points in km.
    Haversine formula:
        a = sin²(Δφ/2) + cos φ1 * cos φ2 * sin²(Δλ/2)
        c = 2 * atan2( √a, √(1−a) )
        d = R * c
    where φ is latitude, λ is longitude, R is earth's radius.

    https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    """
    R = 6371.0088

    orig_lat, orig_lon = np.radians(orig)
    dest_lat, dest_lon = np.radians(dest)

    dlat = (dest_lat-orig_lat)
    dlon = (dest_lon-orig_lon)

    a = np.sin(dlat/2)**2 + np.cos(orig_lat) * np.cos(dest_lat) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    d = d * 1  # --> km
    return d
