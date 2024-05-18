import copy
import cornac
import itertools
import numpy as np
from utils import argnonz, argtopk, rand_member_nb, rand_member_arr_nb, myrecall


SEED = 2024
TOPK = 10
HIDDEN_SIZE = 32


def formatdict(d):
    s = " ".join(f"{k}={v:>7.4f}" if not isinstance(v, str) else f"{k}={v}" for k, v in d.items())
    s = "{" + s + "}"
    return s


# -----------------------------------------------------------------------------


def isOK(y, uid_map, iid_map):
    assert set(np.unique(y)) == {0,1}
    assert y.shape[0] == len(uid_map)
    assert y.shape[1] == len(iid_map)
    assert list(uid_map.keys()) == list(uid_map.values())
    assert list(iid_map.keys()) == list(iid_map.values())
    return True


def prepare_positive_feedback_data(y, uid_map, iid_map):  # --> only positive feedback
    assert isOK(y, uid_map, iid_map)
    N, J = y.shape
    n_implicit_feedback = y.sum()
    uir_tup = (
        np.zeros(n_implicit_feedback, dtype="i8"),  # user_indices
        np.zeros(n_implicit_feedback, dtype="i8"),  # item_indices
        np.zeros(n_implicit_feedback, dtype="f8"),  # feedback values (ratings)
    )
    counter = 0
    for n in range(N):
        for j in range(J):
            if y[n,j] != 0:
                uir_tup[0][counter] = uid_map[n]
                uir_tup[1][counter] = iid_map[j]
                uir_tup[2][counter] = 1
                counter += 1
    assert n_implicit_feedback == counter, "Implicit feedback mismatch detected"
    data = cornac.data.Dataset(num_users=N, num_items=J, uid_map=uid_map, iid_map=iid_map, uir_tuple=uir_tup, seed=SEED)
    return data


def prepare_and_split_positive_feedback_data(y, uid_map, iid_map, te_frac=0.2):  # --> w\i user train-test items split
    assert isOK(y, uid_map, iid_map)
    assert (0 < te_frac < 1)
    N = len(y)
    tr_y = np.zeros(y.shape, dtype="i8")
    te_y = np.zeros(y.shape, dtype="i8")
    tr_frac = 1 - te_frac
    for n in range(N):
        pos = argnonz(y[n,:])
        tr_pos = rand_member_arr_nb(pos, size=max(1, int(len(pos) * tr_frac)))
        te_pos = np.setdiff1d(pos, tr_pos)
        tr_y[n,tr_pos] = 1
        te_y[n,te_pos] = 1
    return (
        prepare_positive_feedback_data(tr_y, uid_map, iid_map),
        prepare_positive_feedback_data(te_y, uid_map, iid_map),
    )


# -----------------------------------------------------------------------------


def hsearch(
        model: cornac.models.Recommender, space: dict[str,np.ndarray|list], space_tied_constraints: dict[str,str],
        tr_data: cornac.data.Dataset,
        te_data: cornac.data.Dataset,
        gridsearch: bool = True, n_iters: int = 0,
):

    assert (tr_data.num_users == te_data.num_users) and (tr_data.uid_map == te_data.uid_map), "Shape or Map mismatch"
    assert (tr_data.num_items == te_data.num_items) and (tr_data.iid_map == te_data.iid_map), "Shape or Map mismatch"

    # grid search hyperparameters optimisation or not?
    __inp = copy.deepcopy(space)
    if gridsearch:
        space = (dict(zip(__inp.keys(), values)) for values in itertools.product(*__inp.values()))
    else:
        space = (__inp)

    # init optimisation state
    best_params = dict()
    best_recsys = None
    best_recall = 0
    tr_data_csr = tr_data.csr_matrix  # for fast row access
    te_data_csr = te_data.csr_matrix  # for fast row access

    acc = 0
    while 1:
        acc += 1

        # sample hyperparameters dict
        if gridsearch:
            try:
                params = next(space)
            except StopIteration:
                break
        else:
            if acc >= n_iters:
                break
            params = {k: rand_member_nb(space[k]) for k in space.keys()}

        # update tied hyperparameters
        for k, k_tied in space_tied_constraints.items():
            params[k] = params[k_tied]

        # optimise
        recsys = model.clone(params)
        recsys.fit(tr_data, te_data)
        # estimate performance for each user in test data
        Q = []
        scores = np.zeros(tr_data.num_items, dtype="f8")
        for n in range(te_data.num_users):
            tr_cn_true = tr_data_csr.getrow(n).indices.astype("i8")
            te_cn_true = te_data_csr.getrow(n).indices.astype("i8")
            # get scores and down weight already chosen train items
            scores[:] = np.squeeze(recsys.score(n))
            scores[tr_cn_true] = -100000
            # calculate test items
            te_cn_pred = argtopk(scores, k=max(TOPK, len(te_cn_true)))
            Q.append(myrecall(te_cn_true, te_cn_pred))

        if (recall := np.mean(Q)) > best_recall:
            best_params = params
            best_recsys = recsys
            best_recall = recall
        del recsys, Q
        print(f"recall={recall:.4f}  "
              f"params={formatdict(params)}  |  best_recall={best_recall:.4f}  best_params={formatdict(best_params)}")

    return best_params, best_recsys, best_recall


# -----------------------------------------------------------------------------


_pop = cornac.models.MostPop(
    name="POP"
)
_pop_search_space = {"name": ["POP"]}
_pop_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_wmf = cornac.models.WMF(
    name="WMF",
    k=HIDDEN_SIZE,
    lambda_u=0.01,
    lambda_v=0.01,
    a=1,                     # the confidence (c_nj) of collected ratings a.k.a. positive feedback
    b=0.01,                  # the confidence (c_nj) of unseen ratings
    learning_rate=1e-3,
    batch_size=96,           # how many random unique items to consider in one batch (num_users x batch_size)
    max_iter=1000,
    verbose=0,
    seed=SEED,
)

_wmf_search_space = {
    "lambda_u"     : np.logspace(-2.0, 2.0, num=10, base=10),
    "b"            : np.logspace(-4.0, 0.0, num=10, base=10),
}
_wmf_search_space_tied_constraints = {"lambda_v": "lambda_u"}  # tied parameter


# -----------------------------------------------------------------------------


_bpr = cornac.models.BPR(
    name="BPR",
    k=HIDDEN_SIZE,
    use_bias=False,
    lambda_reg=0.01,
    learning_rate=1e-3,
    max_iter=8000,           # should be linearly proportional to a nnz in data
    verbose=0,
    num_threads=1,
    seed=SEED,
)

_bpr_search_space = {"lambda_reg": np.logspace(-4.0, 0.0, num=20, base=10)}
_bpr_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_vae = cornac.models.VAECF(
    name="VAE",
    k=HIDDEN_SIZE,
    autoencoder_structure=[HIDDEN_SIZE],
    act_fn="tanh",
    likelihood="mult",
    n_epochs=1000,
    batch_size=256,          # how many random unique users to consider in one batch (batch_size x num_items)
    learning_rate=1e-3,
    beta=1.0,
    verbose=0,
    seed=SEED,
)

_vae_search_space = {"beta": np.linspace(0.5, 1.5, num=20)}
_vae_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_ease = cornac.models.EASE(
    name="EASE",
    lamb=1.0,
    posB=False,
    verbose=0,
    seed=SEED,
)

_ease_search_space = {"lamb": np.logspace(2.0, 2.9, num=20, base=10)}
_ease_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_ngcf = cornac.models.NGCF(
    name="NGCF",
    emb_size=HIDDEN_SIZE,
    layer_sizes=[HIDDEN_SIZE],
    dropout_rates=[0.1],
    num_epochs=250,
    learning_rate=1e-3,
    batch_size=512,
    early_stopping=None,
    lambda_reg=0.01,
    verbose=0,
    seed=SEED,
)

_ngcf_search_space = {"lambda_reg": np.logspace(-4.0, 0.0, num=20, base=10)}
_ngcf_search_space_tied_constraints = {}
