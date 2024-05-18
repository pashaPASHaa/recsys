import argparse
import numpy as np

from numba import njit, i8, f8
from datafactory import load_util_artefacts, dump_aset_artefacts
from utils import argnonz, argtopk, myrecall, mysoftmax, rand_choice_nb, seedutils


"""
    Readings, this module is based on:
    ----------------------------------
    
    [1991] Development and Testing of a Model of Consideration Set Composition
        John H. Roberts and James M. Lattin

    [1995] Studying Consideration Effects in Empirical Choice Models Using Scanner Panel Data
        Rick L. Andrews and T. C. Srinivasan

    [1996] Limited Choice Sets, Local Price Response and Implied Measures of Price Competition
        Bart J. Bronnenberg and Wilfried R. Vanhonacker

    [2009] Blockbuster Culture's Next Rise or Fall: The Impact of Recommender Systems on Sales Diversity
        Daniel M. Fleder and Kartik Hosanagar
"""


SAMPLE_GRID = (4, 8, 12, 16, 20)


@njit((i8[:,::1], f8[:,::1], i8[:,::1], f8, f8))
def fit_beta(y, u, A, lr, eps):

    N, J = y.shape
    orig = 8 * np.ones(N, dtype="f8")
    beta = 8 * np.ones(N, dtype="f8")  # start from large values
    step = 0
    while 1:
        llik = 0
        for n in range(N):
            g = 0
            a = 0
            for j in range(J):
                g += A[n,j] * np.exp(beta[n]*u[n,j]) * u[n,j]
                a += A[n,j] * np.exp(beta[n]*u[n,j])
            grad = 0
            for j in range(J):
                grad += y[n,j] * (u[n,j] - g/a)
                llik += y[n,j] * (beta[n]*u[n,j] - np.log(a))
            beta[n] += lr*grad
            beta[n] = max(0.0, beta[n])
            beta[n] = min(8.0, beta[n])
        if (step % 500 == 0):
            print(llik/N)
        if (np.mean(np.abs(orig-beta)) < eps):
            break
        orig[:] = beta
        step += 1
    return beta


def check_data(y, u, A, eps=1e-8):

    N, J = y.shape
    assert ((N, J) == u.shape), "Shape mismatch"
    assert ((N, J) == A.shape), "Shape mismatch"
    assert set(y.ravel()) == {0,1}, "Bad y"
    assert set(A.ravel()) == {0,1}, "Bad A"
    assert np.all(0-eps <= u) and np.all(u <= 1+eps), "Value outside [0,1] range for utility"
    assert np.all((A*y) == y), "A*y != y"

    # iterate over all users and collect their choices
    recall_at_A = np.zeros(N, dtype="f8")
    recall_at_J = np.zeros(N, dtype="f8")
    UI_mat = np.zeros((N, J), dtype="i8")

    for n in range(N):
        # number of choices
        tn = y[n,:].sum()
        # true choices
        cn = argnonz(y[n,:])
        # make choices un|A
        cn_at_A = argtopk(np.where(A[n,:] == 1, u[n,:], -np.inf), k=tn)
        # make choices un|J
        cn_at_J = argtopk(u[n,:], k=tn)
        # update buffers
        recall_at_A[n] = myrecall(cn, cn_at_A)
        recall_at_J[n] = myrecall(cn, cn_at_J)
        UI_mat[n,cn_at_A] = 1

    Q = np.sum(y     , axis=0)  # observed in population choices
    P = np.sum(UI_mat, axis=0)  # predicted choices
    q = Q/Q.sum()
    p = P/P.sum()

    chisq = np.sum(q*(p/q-1)**2)
    kl_QP = np.sum(q*(np.log(q+eps) - np.log(p+eps)))
    kl_PQ = np.sum(p*(np.log(p+eps) - np.log(q+eps)))

    print(f"Estimated over a population of {N} users KL: QP_loss={kl_QP:.6f} PQ_loss={kl_PQ:.6f} chi2_loss={chisq:.6f}")
    print(f"Recall|A={np.mean(recall_at_A):.4f} sd={np.std(recall_at_A):.4f}  ")
    print(f"Recall|J={np.mean(recall_at_J):.4f} sd={np.std(recall_at_J):.4f}  ")

    return UI_mat


if __name__ == "__main__":

    # -------------------------------------------------------------------------

    # python3 run2.py --help
    parser = argparse.ArgumentParser(description="Synthetic awareness set estimation protocol")
    parser.add_argument("--util_artefacts_file", type=str, required=True, help="file with estimated user preferences")
    parser.add_argument("--aset_artefacts_file", type=str, required=True, help="file to save output")
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    args = parser.parse_args()

    print(f"Detected script configuration for synthetic awareness set estimation protocol:\n"
          f"util_artefacts_file={args.util_artefacts_file} "
          f"aset_artefacts_file={args.aset_artefacts_file} "
          f"seed={args.seed} ")

    # -------------------------------------------------------------------------

    # reproducibility
    np.random.seed(args.seed)
    seedutils(args.seed)
    # load file
    true_data_map, _ = load_util_artefacts(args.util_artefacts_file)
    N, J = true_data_map["y"].shape
    # popularity distribution
    distribution_pop = true_data_map["y"].sum(axis=0) / true_data_map["y"].sum()

    # -------------------------------------------------------------------------

    # init structure for awareness set
    A_map: dict[str,dict[str,dict[str,np.ndarray]]] = {}

    # model awareness
    # estimation is based on a sampling process from a probabilistic model with
    # respect to oracle preferences and multinomial choice behaviour constraint

    for key in true_data_map:

        if key == "y":
            continue
        print(f"\nSet recommendation key={key} as a ground truth of [un]biased preferences")

        y = true_data_map["y"]  # {0,1} observations
        u = true_data_map[key]  # oracle preferences
        A = y.copy()  # init state for awareness set

        A_map[key] = {}
        for k_pop, k_pop_delta in zip(SAMPLE_GRID, np.diff(SAMPLE_GRID, n=1, prepend=0)):
            # populate awareness beyond observed choices
            print(f"k_pop={k_pop}  k_pop_delta={k_pop_delta}")
            for n in range(N):
                for _ in range(k_pop_delta):
                    known = rand_choice_nb(distribution_pop)
                    A[n,known] = 1
            # we need to calibrate choice model based on sampled awareness set
            beta = fit_beta(y, u, A, lr=1e-2, eps=1e-5)
            # check and update
            check_data(y, u, A)
            print(beta)
            A_map[key][str(k_pop)] = {"A": A.copy(), "beta": beta.copy()}

            # model substitution set
            # estimation is based on the complement of awareness set with superior utility properties
            q = [0.05, 0.25, 0.5, 0.75, 0.95]
            p = 0*u
            for n in range(N):
                p[n,:] = mysoftmax(beta[n] * u[n,:])
            print(f"quantiles of interest={q}")
            print(f"A: {np.quantile(np.sum(A, axis=1), q=q)}")
            print(f"y: {np.quantile(np.sum(y, axis=1), q=q)}")
            S = np.quantile(np.sum(p >= np.max(p*y, axis=1, keepdims=True), where=(A == 0), axis=1), q=q)
            print(f"S[probability of consideration > p_choice max]:\n{S}\n")

    # -------------------------------------------------------------------------

    # dump estimated awareness set artefacts in file for later usage
    dump_aset_artefacts(args.aset_artefacts_file, y=true_data_map["y"], A_map=A_map)

    # -------------------------------------------------------------------------

    print(f"DONE!")
