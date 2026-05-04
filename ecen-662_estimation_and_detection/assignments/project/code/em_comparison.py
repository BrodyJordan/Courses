"""
em_comparison.py
================
Compare four Expectation--Maximization variants on a Gaussian Hidden
Markov Model.  The ground-truth HMM is re-sampled from a prior on
every seed so that comparisons aggregate across many problem instances
rather than a single fixed scenario.

Variants implemented (all follow the canonical E-step / M-step template
of [Liu, ECEN 662 Lecture 8] and differ only in how the E-step or the
M-step is handled):

    1. Base EM (Baum--Welch)       : exact forward--backward + closed-form M-step.
    2. Variational EM (VEM)        : mean-field q(z) + closed-form M-step,
                                     following the CAVI rule of [Liu, Lecture 13].
    3. Expectation conditional max : exact E-step + blocked CM-steps with
       (ECM)                         a refreshed posterior between blocks.
    4. Generalized EM (GEM)        : exact E-step + damped (non-maximal)
                                     update toward the closed-form M-step.

Outputs are CSV tables in the script's directory; the LaTeX report
loads them via pgfplots/TikZ.

Tables produced
---------------
    convergence_iter.csv : excess log-likelihood (median + IQR) vs. EM iter.
    mu_error.csv         : aligned mu L1 error vs. EM iter.
    dist_to_truth.csv    : composite distribution distance vs. EM iter.
    complexity.csv       : per-iteration wall time as a function of T.
    density_curves.csv   : recovered marginal densities for one seed.
    density_hist.csv     : histogram of the seed's observations.
    em_summary.csv       : final-iteration summary statistics per method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


METHOD_ORDER = ["Base EM", "VEM", "ECM", "GEM"]
ABBREV = {"Base EM": "base", "VEM": "vem", "ECM": "ecm", "GEM": "gem"}


# ---------------------------------------------------------------------------
# Gaussian HMM core
# ---------------------------------------------------------------------------
@dataclass
class HMMParams:
    """Gaussian HMM parameters stored in log-space where appropriate."""
    log_pi: np.ndarray
    log_A:  np.ndarray
    mu:     np.ndarray
    sigma:  np.ndarray

    @property
    def K(self) -> int:
        return self.mu.shape[0]

    def copy(self) -> "HMMParams":
        return HMMParams(
            self.log_pi.copy(), self.log_A.copy(),
            self.mu.copy(),     self.sigma.copy(),
        )


def emission_log_prob(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """log N(x_t; mu_k, sigma_k^2) for every (t, k). Returns shape (T, K)."""
    x = x.reshape(-1, 1)
    return (-0.5 * np.log(2.0 * np.pi)
            - np.log(sigma)
            - 0.5 * ((x - mu) / sigma) ** 2)


def forward_backward(x: np.ndarray, p: HMMParams):
    """Exact log-space forward--backward recursions ([Liu08])."""
    T, K = x.shape[0], p.K
    log_B = emission_log_prob(x, p.mu, p.sigma)

    log_alpha = np.empty((T, K))
    log_alpha[0] = p.log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + p.log_A, axis=0) + log_B[t]
    log_lik = logsumexp(log_alpha[-1])

    log_beta = np.empty((T, K))
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(p.log_A + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1)

    log_gamma = log_alpha + log_beta - log_lik
    gamma = np.exp(log_gamma)

    log_xi = (log_alpha[:-1, :, None]
              + p.log_A[None, :, :]
              + log_B[1:, None, :]
              + log_beta[1:, None, :]
              - log_lik)
    xi = np.exp(log_xi)

    return gamma, xi, log_lik


def m_step_closed_form(x: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> HMMParams:
    """Closed-form Baum--Welch M-step (Eq.~\\eqref{eq:bw-mstep} in the report)."""
    eps = 1e-12
    pi = gamma[0] + eps
    pi /= pi.sum()
    A = xi.sum(axis=0) + eps
    A /= A.sum(axis=1, keepdims=True)
    weights = gamma.sum(axis=0) + eps
    mu = (gamma * x[:, None]).sum(axis=0) / weights
    var = (gamma * (x[:, None] - mu) ** 2).sum(axis=0) / weights
    sigma = np.sqrt(np.maximum(var, 1e-6))
    return HMMParams(np.log(pi), np.log(A), mu, sigma)


def log_likelihood(x: np.ndarray, p: HMMParams) -> float:
    return forward_backward(x, p)[2]


# ---------------------------------------------------------------------------
# 1. Base EM (Baum--Welch)
# ---------------------------------------------------------------------------
def base_em_step(x: np.ndarray, p: HMMParams, state=None):
    gamma, xi, ll = forward_backward(x, p)
    return m_step_closed_form(x, gamma, xi), ll, None


# ---------------------------------------------------------------------------
# 2. Variational EM (mean-field q(z))
# ---------------------------------------------------------------------------
def cavi_mean_field(x, p, q_init=None, n_sweeps=25, tol=1e-6):
    """CAVI sweep for mean-field q(z) = prod_t q_t(z_t) ([Liu13])."""
    T, K = x.shape[0], p.K
    log_B = emission_log_prob(x, p.mu, p.sigma)
    q = np.full((T, K), 1.0 / K) if q_init is None else q_init.copy()

    elbo_prev = -np.inf
    for _ in range(n_sweeps):
        for t in range(T):
            log_qt = log_B[t].copy()
            if t == 0:
                log_qt = log_qt + p.log_pi
            else:
                log_qt = log_qt + q[t - 1] @ p.log_A
            if t < T - 1:
                log_qt = log_qt + q[t + 1] @ p.log_A.T
            log_qt = log_qt - logsumexp(log_qt)
            q[t] = np.exp(log_qt)
        elbo = mean_field_elbo(x, p, q)
        if abs(elbo - elbo_prev) < tol * (abs(elbo_prev) + 1.0):
            break
        elbo_prev = elbo
    xi_mf = q[:-1, :, None] * q[1:, None, :]
    return q, xi_mf, elbo


def mean_field_elbo(x, p, q):
    log_B = emission_log_prob(x, p.mu, p.sigma)
    e_log_p = (q[0] * p.log_pi).sum()
    e_log_p += np.sum(q[:-1, :, None] * q[1:, None, :] * p.log_A[None, :, :])
    e_log_p += np.sum(q * log_B)
    h_q = -np.sum(q * np.log(q + 1e-12))
    return float(e_log_p + h_q)


def vem_step(x, p, state=None):
    q_init = state["q"] if state is not None else None
    q, xi_mf, _ = cavi_mean_field(x, p, q_init=q_init, n_sweeps=20)
    new_p = m_step_closed_form(x, q, xi_mf)
    ll = log_likelihood(x, new_p)
    return new_p, ll, {"q": q}


# ---------------------------------------------------------------------------
# 3. Expectation Conditional Maximization (ECM)
# ---------------------------------------------------------------------------
def ecm_step(x, p, state=None):
    """ECM with an E-step refresh between dynamics and emission CM-steps."""
    eps = 1e-12

    gamma1, xi1, ll = forward_backward(x, p)
    pi = gamma1[0] + eps
    pi /= pi.sum()
    A = xi1.sum(axis=0) + eps
    A /= A.sum(axis=1, keepdims=True)
    log_pi, log_A = np.log(pi), np.log(A)

    intermediate = HMMParams(log_pi, log_A, p.mu.copy(), p.sigma.copy())
    gamma2, _, _ = forward_backward(x, intermediate)

    weights = gamma2.sum(axis=0) + eps
    mu = (gamma2 * x[:, None]).sum(axis=0) / weights
    var = (gamma2 * (x[:, None] - mu) ** 2).sum(axis=0) / weights
    sigma = np.sqrt(np.maximum(var, 1e-6))
    return HMMParams(log_pi, log_A, mu, sigma), ll, None


# ---------------------------------------------------------------------------
# 4. Generalized EM (damped step toward the closed-form M-step)
# ---------------------------------------------------------------------------
def gem_step(x, p, state=None, alpha=0.5):
    gamma, xi, ll = forward_backward(x, p)
    full = m_step_closed_form(x, gamma, xi)

    log_pi = (1.0 - alpha) * p.log_pi + alpha * full.log_pi
    log_pi -= logsumexp(log_pi)
    log_A = (1.0 - alpha) * p.log_A + alpha * full.log_A
    log_A -= logsumexp(log_A, axis=1, keepdims=True)
    mu = (1.0 - alpha) * p.mu + alpha * full.mu
    log_sigma = (1.0 - alpha) * np.log(p.sigma) + alpha * np.log(full.sigma)
    sigma = np.exp(log_sigma)
    return HMMParams(log_pi, log_A, mu, sigma), ll, None


METHODS = {
    "Base EM": base_em_step,
    "VEM":     vem_step,
    "ECM":     ecm_step,
    "GEM":     gem_step,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_em(x, init, step_fn, max_iter=80):
    p = init.copy()
    history = {
        "log_lik":   [log_likelihood(x, p)],
        "wall_time": [0.0],
        "params":    [p.copy()],
    }
    state = None
    t_start = time.perf_counter()
    for _ in range(max_iter):
        p, _, state = step_fn(x, p, state=state)
        history["wall_time"].append(time.perf_counter() - t_start)
        history["log_lik"].append(log_likelihood(x, p))
        history["params"].append(p.copy())
    history["log_lik"]   = np.asarray(history["log_lik"])
    history["wall_time"] = np.asarray(history["wall_time"])
    return p, history


# ---------------------------------------------------------------------------
# Random ground-truth sampling
# ---------------------------------------------------------------------------
def sample_true_params(rng, K=3, min_mu_gap: float = 0.5) -> HMMParams:
    """Draw a random Gaussian HMM from broad priors.

    - Initial-state distribution :math:`\\pi^\\star \\sim Dirichlet(1)`
      (uniform over the simplex).
    - Each row of :math:`A^\\star` is an independent
      :math:`Dirichlet(\\mathbf{1} + d_j e_j)` sample, where the
      diagonal concentration :math:`d_j \\sim Exp(2)` varies the chain
      from near-uniform to very sticky.
    - Component means are i.i.d. :math:`\\mathcal{N}(0, 9)` and sorted;
      draws with adjacent gap below ``min_mu_gap`` are rejected to keep
      the mixture nominally identifiable.
    - Component standard deviations are i.i.d. :math:`Uniform(0.3, 1.5)`.
    """
    pi = rng.dirichlet(np.ones(K))

    A = np.empty((K, K))
    for j in range(K):
        diag_extra = rng.exponential(2.0)
        alpha = np.ones(K) + diag_extra * (np.arange(K) == j)
        A[j] = rng.dirichlet(alpha)

    while True:
        mu = np.sort(rng.normal(0.0, 3.0, size=K))
        if np.min(np.diff(mu)) >= min_mu_gap:
            break

    sigma = rng.uniform(0.3, 1.5, size=K)

    return HMMParams(np.log(pi), np.log(A), mu, sigma)


def sample_hmm(rng, T, p: HMMParams):
    K = p.K
    A = np.exp(p.log_A)
    pi = np.exp(p.log_pi)
    z = np.empty(T, dtype=int)
    z[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        z[t] = rng.choice(K, p=A[z[t - 1]])
    x = p.mu[z] + p.sigma[z] * rng.standard_normal(T)
    return x, z


def make_init(rng, x, K, perturb=0.4) -> HMMParams:
    quantiles = np.quantile(x, np.linspace(0.15, 0.85, K))
    mu = quantiles + perturb * rng.standard_normal(K)
    sigma = np.full(K, max(x.std(), 0.5))
    log_pi = np.full(K, -np.log(K))
    log_A = np.log(np.full((K, K), 1.0 / K)) + 0.05 * rng.standard_normal((K, K))
    log_A -= logsumexp(log_A, axis=1, keepdims=True)
    return HMMParams(log_pi, log_A, mu, sigma)


# ---------------------------------------------------------------------------
# Distance to truth
# ---------------------------------------------------------------------------
def hungarian_perm(mu_est, mu_true) -> np.ndarray:
    """Return the permutation that aligns mu_est with mu_true."""
    cost = np.abs(mu_est[:, None] - mu_true[None, :])
    row, col = linear_sum_assignment(cost)
    perm = np.empty_like(row)
    perm[col] = row
    return perm


def best_perm_mu_error(mu_est, mu_true) -> float:
    cost = np.abs(mu_est[:, None] - mu_true[None, :])
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def _gauss_sym_kl(mu_a, s_a, mu_b, s_b) -> np.ndarray:
    """Symmetric KL between N(mu_a, s_a^2) and N(mu_b, s_b^2), elementwise."""
    kl_ab = np.log(s_b / s_a) + (s_a ** 2 + (mu_a - mu_b) ** 2) / (2 * s_b ** 2) - 0.5
    kl_ba = np.log(s_a / s_b) + (s_b ** 2 + (mu_a - mu_b) ** 2) / (2 * s_a ** 2) - 0.5
    return 0.5 * (kl_ab + kl_ba)


def stationary_dist(A: np.ndarray, n_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix A via power iteration."""
    K = A.shape[0]
    pi = np.full(K, 1.0 / K)
    for _ in range(n_iter):
        pi_new = pi @ A
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi / pi.sum()


def marginal_pdf(x_grid: np.ndarray, p: HMMParams) -> np.ndarray:
    """Stationary marginal density of X_t under the HMM p:

        p(x) = sum_k pi_inf_k * N(x; mu_k, sigma_k^2),

    where pi_inf is the stationary distribution of A.
    """
    pi_inf = stationary_dist(np.exp(p.log_A))
    norm = (np.exp(-0.5 * ((x_grid[:, None] - p.mu) / p.sigma) ** 2)
            / (np.sqrt(2.0 * np.pi) * p.sigma))
    return (norm * pi_inf).sum(axis=1)


def distribution_distance(p_est: HMMParams, p_true: HMMParams) -> float:
    """Composite distance between two Gaussian HMMs.

    After Hungarian alignment of the emission components, returns the
    average symmetric KL across emission states plus the Frobenius
    distance between transition matrices and the TV distance between
    initial-state distributions (all scaled to a common O(1) range).
    """
    perm = hungarian_perm(p_est.mu, p_true.mu)
    mu_a   = p_est.mu[perm]
    sig_a  = p_est.sigma[perm]
    pi_a   = np.exp(p_est.log_pi)[perm]
    A_a    = np.exp(p_est.log_A)[np.ix_(perm, perm)]

    emission_kl = _gauss_sym_kl(mu_a, sig_a, p_true.mu, p_true.sigma).mean()
    transition_err = np.linalg.norm(A_a - np.exp(p_true.log_A))
    initial_tv = 0.5 * np.abs(pi_a - np.exp(p_true.log_pi)).sum()
    return float(emission_kl + transition_err + initial_tv)


# ---------------------------------------------------------------------------
# Convergence experiment over many seeds, with a fresh truth per seed
# ---------------------------------------------------------------------------
def convergence_experiment(seeds, T=400, K=3, max_iter=80):
    results = {name: {"ll_excess": [], "mu_err": [], "dist": []}
               for name in METHODS}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        truth = sample_true_params(rng, K=K)
        x, _ = sample_hmm(rng, T, truth)
        ll_truth = log_likelihood(x, truth)
        init = make_init(rng, x, K)

        for name, step_fn in METHODS.items():
            _, hist = run_em(x, init, step_fn, max_iter=max_iter)
            ll_excess = hist["log_lik"] - ll_truth
            mu_err = np.array([best_perm_mu_error(p.mu, truth.mu)
                               for p in hist["params"]])
            dist = np.array([distribution_distance(p, truth)
                             for p in hist["params"]])
            results[name]["ll_excess"].append(ll_excess)
            results[name]["mu_err"].append(mu_err)
            results[name]["dist"].append(dist)

    for name in results:
        for key in ("ll_excess", "mu_err", "dist"):
            results[name][key] = np.stack(results[name][key])

    return results, T, K


# ---------------------------------------------------------------------------
# Complexity-vs-T experiment (still with random truth per seed)
# ---------------------------------------------------------------------------
def complexity_experiment(T_values, K=3, n_seeds=4, n_iter=6):
    out = {name: {T: [] for T in T_values} for name in METHODS}
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        truth = sample_true_params(rng, K=K)
        for T in T_values:
            x, _ = sample_hmm(rng, T, truth)
            init = make_init(rng, x, K)
            for name, step_fn in METHODS.items():
                p = init.copy()
                state = None
                p, _, state = step_fn(x, p, state=state)  # warm start
                t0 = time.perf_counter()
                for _ in range(n_iter):
                    p, _, state = step_fn(x, p, state=state)
                out[name][T].append((time.perf_counter() - t0) / n_iter)
    return out


# ---------------------------------------------------------------------------
# CSV writers (consumed by pgfplots / TikZ in the report)
# ---------------------------------------------------------------------------
def _stat_columns():
    cols = []
    for name in METHOD_ORDER:
        for stat in ("p25", "med", "p75"):
            cols.append(f"{ABBREV[name]}_{stat}")
    return cols


def _band_row(values_per_method):
    cells = []
    for name in METHOD_ORDER:
        arr = values_per_method[name]
        cells.append(f"{np.percentile(arr, 25):.6f}")
        cells.append(f"{np.median(arr):.6f}")
        cells.append(f"{np.percentile(arr, 75):.6f}")
    return cells


def write_iter_table(out_path, x_label, x_values, results, key):
    header = ",".join([x_label, *_stat_columns()])
    rows = [header]
    for i, x in enumerate(x_values):
        cells = [f"{x}"] + _band_row({m: results[m][key][:, i] for m in METHODS})
        rows.append(",".join(cells))
    out_path.write_text("\n".join(rows) + "\n")


def write_complexity(complexity, out_dir: Path):
    Ts = sorted(complexity[METHOD_ORDER[0]].keys())
    header = ",".join(["T", *_stat_columns()])
    rows = [header]
    for T in Ts:
        cells = [str(T)]
        for name in METHOD_ORDER:
            arr = np.asarray(complexity[name][T]) * 1e3
            cells.append(f"{np.percentile(arr, 25):.6f}")
            cells.append(f"{np.median(arr):.6f}")
            cells.append(f"{np.percentile(arr, 75):.6f}")
        rows.append(",".join(cells))
    (out_dir / "complexity.csv").write_text("\n".join(rows) + "\n")


def write_density_comparison(seed: int, out_dir: Path,
                             T: int = 400, K: int = 3,
                             max_iter: int = 80, n_bins: int = 30):
    """For one seed, run all methods and save the recovered marginal
    densities alongside the truth and a histogram of the observations.

    Outputs
    -------
    density_curves.csv : x grid, truth_pdf, base_pdf, vem_pdf, ecm_pdf, gem_pdf.
    density_hist.csv   : (N+1) bin edges and N densities (padded with one
                          trailing zero) in the format expected by
                          pgfplots' ``ybar interval'' style.
    """
    rng = np.random.default_rng(seed)
    truth = sample_true_params(rng, K=K)
    x, _ = sample_hmm(rng, T, truth)
    init = make_init(rng, x, K)

    final = {"truth": truth}
    for name, step_fn in METHODS.items():
        p_final, _ = run_em(x, init, step_fn, max_iter=max_iter)
        final[name] = p_final

    x_min, x_max = float(x.min()) - 1.0, float(x.max()) + 1.0
    grid = np.linspace(x_min, x_max, 300)
    truth_pdf = marginal_pdf(grid, final["truth"])
    method_pdfs = {name: marginal_pdf(grid, final[name]) for name in METHOD_ORDER}

    cols = ["x", "truth_pdf"] + [f"{ABBREV[n]}_pdf" for n in METHOD_ORDER]
    rows = [",".join(cols)]
    for i, xg in enumerate(grid):
        cells = [f"{xg:.6f}", f"{truth_pdf[i]:.6f}"]
        for name in METHOD_ORDER:
            cells.append(f"{method_pdfs[name][i]:.6f}")
        rows.append(",".join(cells))
    (out_dir / "density_curves.csv").write_text("\n".join(rows) + "\n")

    counts, edges = np.histogram(x, bins=n_bins, density=True)
    hist_rows = ["x,density"]
    for i, e in enumerate(edges):
        d = counts[i] if i < len(counts) else 0.0
        hist_rows.append(f"{e:.6f},{d:.6f}")
    (out_dir / "density_hist.csv").write_text("\n".join(hist_rows) + "\n")


def write_summary(results, out_dir: Path):
    rows = ["method,final_ll_excess_mean,final_ll_excess_std,"
            "final_dist_mean,final_dist_std,iters"]
    for name in METHOD_ORDER:
        ll  = results[name]["ll_excess"][:, -1]
        d   = results[name]["dist"][:, -1]
        n_iter = results[name]["ll_excess"].shape[1] - 1
        rows.append(
            f"{ABBREV[name]},{ll.mean():.4f},{ll.std():.4f},"
            f"{d.mean():.4f},{d.std():.4f},{n_iter}"
        )
    (out_dir / "em_summary.csv").write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(20))
    print(f"Running convergence experiment on {len(seeds)} seeds (random truth per seed) ...")
    results, T, K = convergence_experiment(seeds, T=400, K=3, max_iter=80)

    n_iter = results["Base EM"]["ll_excess"].shape[1]
    iters  = np.arange(n_iter)
    write_iter_table(out_dir / "convergence_iter.csv", "iter", iters, results, "ll_excess")
    write_iter_table(out_dir / "mu_error.csv",         "iter", iters, results, "mu_err")
    write_iter_table(out_dir / "dist_to_truth.csv",    "iter", iters, results, "dist")
    write_summary(results, out_dir)

    T_values = [100, 200, 400, 800, 1600]
    print(f"Running complexity experiment for T in {T_values} ...")
    complexity = complexity_experiment(T_values, K=3, n_seeds=4, n_iter=5)
    write_complexity(complexity, out_dir)

    print("Generating density-comparison tables for seed 0 ...")
    write_density_comparison(seed=0, out_dir=out_dir, T=400, K=3, max_iter=80)

    print(f"Done. Wrote tables to {out_dir}:")
    for f in (
        "convergence_iter.csv", "mu_error.csv", "dist_to_truth.csv",
        "complexity.csv", "density_curves.csv", "density_hist.csv",
        "em_summary.csv",
    ):
        print(f"  {f}")


if __name__ == "__main__":
    main()
