"""
basket_cds_engine

The below code covers:
- CDS hazard-rate calibration
- default-time simulation
- Gaussian and Student-t copulas
- t-copula calibration
- k-th-to-default pricing
- pricing checks

Most of the important functions are kept here so that the calculations used in the notebook
can be reproduced.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t, chi2, genpareto, kendalltau, spearmanr
from scipy.stats import qmc
from scipy.special import gammaln
from scipy.optimize import brentq, minimize_scalar
from scipy.linalg import cholesky, eigh, cho_factor, cho_solve

__all__ = [
    "protection_leg_identity",
    "nearest_pd_correlation", "kendall_to_linear", "spearman_to_linear",
    "survival_from_hazards", "bootstrap_hazards", "make_default_time_inverse",
    "base_uniforms", "gaussian_copula", "t_copula",
    "t_copula_logpdf", "profile_loglik_nu", "fit_nu_mle",
    "kde_cdf", "semiparametric_cdf", "pseudo_samples",
    "price_basket", "single_name_spread",
    "tail_dependence_empirical", "tail_dependence_theoretical",
    "validate_ladder",
]


# correlation utilities


def nearest_pd_correlation(A, eps=1e-8):
    """Project a symmetric matrix onto the nearest positive-definite correlation
    matrix by eigenvalue flooring, then re-normalise the diagonal to 1.
    """
    A = np.asarray(A, float)
    A = (A + A.T) / 2.0
    w, V = eigh(A)
    min_eig = w.min()
    w = np.clip(w, eps, None)
    A_pd = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A_pd))
    A_pd = A_pd / np.outer(d, d)
    np.fill_diagonal(A_pd, 1.0)
    return A_pd, min_eig


def kendall_to_linear(tau):
    """Convert Kendall's tau to linear correlation using the sine transformation"""
    M = np.sin(np.pi / 2.0 * np.asarray(tau, float))
    np.fill_diagonal(M, 1.0)
    return M


def spearman_to_linear(rho_s):
    """Convert Spearman's rho to linear correlation"""
    M = 2.0 * np.sin(np.pi / 6.0 * np.asarray(rho_s, float))
    np.fill_diagonal(M, 1.0)
    return M



# CDS hazard-rate calibration


def survival_from_hazards(edges, hazards):
    """    Build a survival function from piecewise-constant hazard rates.
    This creates a callable Q(t) that gives the survival probability at time t.
    The hazards are constant between the given edges, which is the standard
    market convention for CDS curves."""
    edges = np.asarray(edges, float)
    hazards = np.asarray(hazards, float)
    cum = np.concatenate([[0.0], np.cumsum(hazards * np.diff(edges))])

    def Q(t):
        t = np.asarray(t, float)
        k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(hazards) - 1)
        return np.exp(-(cum[k] + hazards[k] * (t - edges[k])))
    return Q


def bootstrap_hazards(tenors, spreads_bps, recovery, discount, prem_freq=4):
    """    Strip piecewise-constant hazards from CDS spreads.
    This is the workhorse function. It solves for the hazard rate at each pillar
    such that the CDS prices to par. 
    """
    tenors = np.asarray(tenors, float)
    spreads = np.asarray(spreads_bps, float) / 1e4
    hazards: list[float] = []
    for i, T in enumerate(tenors):
        s_i = spreads[i]
        edges_i = np.concatenate([[0.0], tenors[:i + 1]])

        def par_gap(h_try, s_i=s_i, edges_i=edges_i, T=T):
            hz = np.array(hazards + [h_try])
            Q = survival_from_hazards(edges_i, hz)
            grid = np.arange(1, int(round(T * prem_freq)) + 1) / prem_freq
            dt = 1.0 / prem_freq
            P = discount(grid)
            Qc, Qp = Q(grid), Q(grid - dt)
            protection = (1.0 - recovery) * np.sum(P * (Qp - Qc))
            premium = s_i * np.sum(dt * P * Qc) + s_i * np.sum(0.5 * dt * P * (Qp - Qc))
            return protection - premium

        hazards.append(brentq(par_gap, 1e-9, 5.0, xtol=1e-12))
    return np.array(hazards), np.concatenate([[0.0], tenors])


def single_name_spread(edges, hazards, recovery, discount, T=5.0, prem_freq=4):
    """Reprice a single-name par CDS from a hazard curve.  
    This is mostly used for validation - if you bootstrap a curve and then
    reprice it, you should get back the original spread (within numerical
    tolerance). If not, something's wrong with the bootstrap."""
    Q = survival_from_hazards(edges, hazards)
    grid = np.arange(1, int(round(T * prem_freq)) + 1) / prem_freq
    dt = 1.0 / prem_freq
    P = discount(grid)
    Qc, Qp = Q(grid), Q(grid - dt)
    dl = (1.0 - recovery) * np.sum(P * (Qp - Qc))
    pl01 = np.sum(dt * P * Qc) + np.sum(0.5 * dt * P * (Qp - Qc))
    return 1e4 * dl / pl01



# default-time simulation


def make_default_time_inverse(edges, hazards, cap=40.0):
    """    Build the inverse survival function: tau = Q^{-1}(u).
    This is where we go from uniform random variables to default times.
    The algorithm is vectorized because in practice you're simulating
    hundreds of thousands of paths and performance matters.
    """
    edges = np.asarray(edges, float)
    hazards = np.asarray(hazards, float)
    left = np.append(0.0, edges[1:])
    hz = np.append(hazards, hazards[-1])          # flat extrapolation past last pillar
    widths = np.append(np.diff(edges), cap - edges[-1])
    cum = np.concatenate([[0.0], np.cumsum(hz * widths)])

    def inv(U):
        U = np.clip(np.asarray(U, float), 1e-12, 1.0 - 1e-12)
        L = -np.log(U)                             # cumulative-hazard threshold
        k = np.clip(np.searchsorted(cum, L, side="right") - 1, 0, len(hz) - 1)
        tau = left[k] + (L - cum[k]) / hz[k]
        return np.where(L >= cum[-1], np.inf, tau)
    return inv



# Gaussian and Student-t copulas


def base_uniforms(d, n, engine, rng):
    """To Generate uniform draws from various low-discrepancy sequences.
    MC can be an option, but Sobol sequences give much better convergence
    for the same number of paths. The scrambled versions are important -
    plain Sobol can have structure that biases the results.  
    The Halton sequence is there as an alternative as well.
    """
    if engine == "mc":
        return rng.random((n, d))
    seed = int(rng.integers(1_000_000_000))
    if engine == "sobol":
        s = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        return s.random_base2(m)[:n]
    if engine == "halton":
        return qmc.Halton(d=d, scramble=True, seed=seed).random(n)
    raise ValueError(f"unknown engine {engine!r}")


def gaussian_copula(corr, n, engine="mc", rng=None):
    """The Gaussian copula - simple, fast, and everyone's first choice.
    
    The implementation is standard: draw normal variables, correlate them,
    then transform back to uniform using the normal CDF. The Cholesky
    decomposition is stable as long as the correlation matrix is positive
    definite"""
    rng = np.random.default_rng() if rng is None else rng
    L = cholesky(np.asarray(corr, float), lower=True)
    u = base_uniforms(corr.shape[0], n, engine, rng)
    z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    return norm.cdf(z @ L.T)


def t_copula(corr, nu, n, engine="mc", rng=None):
    """The Student-t copula - for when you need tail dependence.
    This is the engine for basket CDS pricing. The t-copula has fatter tails
    than the Gaussian, which makes a big difference for k-th-to-default spreads.
    The implementation uses a (d+1)-th QMC coordinate for the chi-square draw.
    This matters: if you mix QMC with pseudo-random numbers, you lose the
    low-discrepancy property. Took me a while to figure this out.
    """
    rng = np.random.default_rng() if rng is None else rng
    d = corr.shape[0]
    L = cholesky(np.asarray(corr, float), lower=True)
    u = base_uniforms(d + 1, n, engine, rng)
    z = norm.ppf(np.clip(u[:, :d], 1e-10, 1 - 1e-10))
    x = z @ L.T
    s = chi2.ppf(np.clip(u[:, d], 1e-10, 1 - 1e-10), df=nu)
    y = x / np.sqrt(s / nu)[:, None]
    return student_t.cdf(y, df=nu)



# t-copula calibration


def t_copula_logpdf(U, nu, corr):
    """Log-density of the t-copula.
    This is the likelihood function for the t-copula.
    """
    U = np.clip(np.atleast_2d(np.asarray(U, float)), 1e-12, 1 - 1e-12)
    n = U.shape[1]
    xi = student_t.ppf(U, df=nu)

    c, low = cho_factor(np.asarray(corr, float), lower=True)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    quad = np.einsum("ij,ij->i", xi, cho_solve((c, low), xi.T).T)

    const = (gammaln((nu + n) / 2.0) - gammaln(nu / 2.0)
             + n * (gammaln(nu / 2.0) - gammaln((nu + 1.0) / 2.0)))
    num = -((nu + n) / 2.0) * np.log1p(quad / nu)
    den = ((nu + 1.0) / 2.0) * np.sum(np.log1p(xi ** 2 / nu), axis=1)
    return -0.5 * logdet + const + num + den


def profile_loglik_nu(U, corr, nu_grid):
    """Evaluate the log-likelihood over a grid of nu values.
    This is useful for visualization and for getting a good initial guess
    before refinement."""
    return np.array([t_copula_logpdf(U, float(v), corr).sum() for v in nu_grid])


def fit_nu_mle(U, corr, nu_grid=None, refine=True):
    """Maximum likelihood estimation of the degrees of freedom parameter.
    This is the heart of the t-copula calibration. We first do a coarse grid
    search to get in the right ballpark, then refine using Brent's method."""
    nu_grid = np.arange(2.0, 51.0) if nu_grid is None else np.asarray(nu_grid, float)
    ll = profile_loglik_nu(U, corr, nu_grid)
    nu_hat = float(nu_grid[int(np.argmax(ll))])
    if refine:
        lo, hi = max(2.02, nu_hat - 2.0), nu_hat + 2.0
        res = minimize_scalar(lambda v: -t_copula_logpdf(U, float(v), corr).sum(),
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-3})
        if res.success:
            nu_hat = float(res.x)
    return nu_hat, nu_grid, ll



# pseudo-samples


def kde_cdf(data, bw=None):
    """Kernel density estimate of the CDF.
    Instead of the usual empirical CDF, we use a smoothed version. This avoids
    the zeros and ones that cause problems with the copula transform.
    The bandwidth is defaulted as per the Silverman's.
    """
    x = np.sort(np.asarray(data, float))
    n = x.size
    if bw is None:
        sd = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        scale = min(sd, iqr / 1.349) if iqr > 0 else sd
        bw = 0.9 * scale * n ** (-1 / 5)

    def cdf(v):
        v = np.atleast_1d(np.asarray(v, float))
        out = norm.cdf((v[:, None] - x[None, :]) / bw).mean(axis=1)
        return np.clip(out, 1e-6, 1 - 1e-6)
    return cdf, bw


def semiparametric_cdf(data, lower_q=0.10, upper_q=0.90, bw=None):
    """Semi-parametric CDF with KDE interior and GPD tails.
    This is my preferred approach. The middle of the distribution is 
    smooth and the tails are fitted with a Generalized Pareto Distribution.
    This gives you the best of both worlds: flexibility in the
    center with a theoretically sound tail.
    """
    x = np.sort(np.asarray(data, float))
    lo, hi = np.quantile(x, lower_q), np.quantile(x, upper_q)
    interior, bw = kde_cdf(x, bw=bw)

    low_exc = lo - x[x < lo]
    up_exc = x[x > hi] - hi
    c_lo, _, s_lo = genpareto.fit(low_exc, floc=0) if low_exc.size >= 5 else (0.0, 0, 1.0)
    c_hi, _, s_hi = genpareto.fit(up_exc, floc=0) if up_exc.size >= 5 else (0.0, 0, 1.0)

    def cdf(v):
        v = np.atleast_1d(np.asarray(v, float))
        out = interior(v)
        m_lo = v < lo
        if m_lo.any():
            out[m_lo] = lower_q * (1.0 - genpareto.cdf(lo - v[m_lo], c_lo, scale=s_lo))
        m_hi = v > hi
        if m_hi.any():
            out[m_hi] = upper_q + (1.0 - upper_q) * genpareto.cdf(v[m_hi] - hi, c_hi, scale=s_hi)
        return np.clip(out, 1e-6, 1 - 1e-6)

    par = dict(xi_lo=c_lo, beta_lo=s_lo, xi_hi=c_hi, beta_hi=s_hi, lo=lo, hi=hi, bw=bw)
    return cdf, par


def pseudo_samples(df, method="semiparametric", **kw):
    """Transform data columns into pseudo-samples (uniform margins).
    """
    out = np.empty(df.shape)
    pars = {}
    for j, col in enumerate(df.columns):
        v = df[col].to_numpy(float)
        if method == "semiparametric":
            F, p = semiparametric_cdf(v, **kw)
        elif method == "kde":
            F, bw = kde_cdf(v, **kw)
            p = {"bw": bw}
        elif method == "ecdf":
            r = v.argsort().argsort() + 1
            F, p = (lambda z, r=r, n=len(v): (r - 0.5) / n), {}
            out[:, j] = (r - 0.5) / len(v)
            pars[col] = p
            continue
        else:
            raise ValueError(method)
        out[:, j] = F(v)
        pars[col] = p
    return out, pars



# k-th-to-default pricing


def price_basket(U, inverses, recoveries, discount, T=5.0, prem_freq=4,
                 ks=(1, 2, 3, 4, 5), payoff="brief", notional_reduction=True,
                 tau_floor=0.25, return_paths=False):
    """    Price k-th-to-default basket spreads.
    This is where all the pieces come together. Given simulated uniforms and
    the inverse survival functions, we compute the fair spread for each
    k-th-to-default contract in the basket.
    
    The key design decisions:
    - "brief" payoff scales protection by 1/n (standard for index tranches)
    - Notional reduction removes defaulted names from the premium leg
    - tau_floor prevents instability from very early defaults
    
    The notional_reduction option is important. In the real market, when a
    name defaults, the notional decreases. If you ignore this, you'll overstate
    the premium leg and understate the spread.
    The tau_floor is a practical hack. Sometimes simulations produce default
    times very close to zero, which makes the spread blow up. A floor of
    0.25 years (3 months) is reasonable and prevents numerical issues.
    """
    U = np.asarray(U, float)
    n_paths, n_names = U.shape
    tau = np.empty_like(U)
    for j, inv in enumerate(inverses):
        tau[:, j] = inv(U[:, j])

    tau_sorted = np.sort(tau, axis=1)
    order = np.argsort(tau, axis=1)
    rec_sorted = np.asarray(recoveries, float)[order]

    coupons = np.arange(1, int(round(T * prem_freq)) + 1) / prem_freq
    dt = 1.0 / prem_freq
    P = discount(coupons)                                   # (M,)
    lgd_scale = (1.0 / n_names) if payoff == "brief" else 1.0

    # number of names already defaulted strictly before each coupon date
    n_def = (tau_sorted[:, None, :] < coupons[None, :, None]).sum(axis=2)   # (N, M)

    out, paths = {}, {}
    for k in ks:
        tk = tau_sorted[:, k - 1]
        if tau_floor is not None:
            tk = np.maximum(tk, tau_floor)
        Rk = rec_sorted[:, k - 1]
        hit = np.isfinite(tk) & (tk <= T)
        t_pay = np.where(hit, np.minimum(tk, T), T)

        # -------- default leg --------
        dl = np.where(hit, (1.0 - Rk) * lgd_scale * discount(t_pay), 0.0)

        # -------- premium leg --------
        alive = coupons[None, :] <= tk[:, None]
        if notional_reduction:
            w = np.clip(n_names - n_def, 0, n_names) / n_names          # (N, M)
        else:
            w = np.ones_like(alive, dtype=float)
        annuity = (dt * P[None, :] * alive * w).sum(axis=1)
        # accrual from the last coupon to the trigger (finite paths only)
        tk_safe = np.where(hit, tk, 0.0)
        last = np.floor(tk_safe * prem_freq) / prem_freq
        stub = np.where(hit, np.clip(tk_safe - last, 0.0, dt), 0.0)
        w_stub = np.where(notional_reduction,
                          np.clip(n_names - (k - 1), 0, n_names) / n_names, 1.0)
        annuity = annuity + np.where(hit, stub * discount(t_pay) * w_stub, 0.0)

        dl_bar, pl_bar = dl.mean(), annuity.mean()
        out[k] = {
            "spread_bps": 1e4 * dl_bar / pl_bar,
            "DL": dl_bar, "RPV01": pl_bar,
            "prob_trigger": float(hit.mean()),
            "se_bps": 1e4 * dl.std(ddof=1) / np.sqrt(n_paths) / pl_bar,   # delta-method, DL term
        }
        if return_paths:
            paths[k] = {"tau_k": tk, "hit": hit, "DL": dl, "RPV01": annuity}
    return (out, paths) if return_paths else out



# tail dependence


def tail_dependence_empirical(U, i=0, j=1, q=0.05):
    """Empirical estimate of lower tail dependence.
    """
    a = U[:, i] < q
    if a.sum() == 0:
        return np.nan
    return float((a & (U[:, j] < q)).mean() / a.mean())


def tail_dependence_theoretical(rho, nu):
    """    Theoretical tail dependence for the t-copula.
    This is a closed-form expression for the lower tail dependence coefficient
    of the t-copula. It depends on the correlation and the degrees of freedom.
    For the Gaussian copula, this would be 0 for all rho < 1, which is one
    reason why the t-copula is preferred for credit applications.
    """
    rho = np.asarray(rho, float)
    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * student_t.cdf(arg, df=nu + 1)



# pricing checks


def protection_leg_identity(U, inverses, recoveries, discount, T=5.0,
                            payoff="brief", tau_floor=None):
    """Exact arbitrage identity on the protection legs.
    This is the sharpest sanity check in the whole framework. Every simulated
    default is the k-th default for exactly one k, so the sum of the basket
    protection legs must equal the sum of the individual protection legs.
    
    Unlike spread-level identities, this holds exactly (up to numerical error).
    If this fails, something is fundamentally wrong with the simulation or the
    pricing logic.
    I run this test on every new dataset. It's caught bugs that would have
    otherwise gone unnoticed.
    """
    U = np.asarray(U, float)
    n_names = U.shape[1]
    tau = np.column_stack([inv(U[:, j]) for j, inv in enumerate(inverses)])
    rec = np.asarray(recoveries, float)
    scale = (1.0 / n_names) if payoff == "brief" else 1.0

    # per-name legs
    hit_i = np.isfinite(tau) & (tau <= T)
    dl_names = np.where(hit_i, (1.0 - rec)[None, :] * scale * discount(np.where(hit_i, tau, T)), 0.0)

    # basket legs, summed over k
    res = price_basket(U, inverses, recoveries, discount, T=T, payoff=payoff,
                       tau_floor=tau_floor, ks=tuple(range(1, n_names + 1)))
    lhs = sum(res[k]["DL"] for k in res)
    rhs = dl_names.sum(axis=1).mean()
    return lhs, rhs, abs(lhs - rhs) / rhs


def validate_ladder(res, single_spreads, payoff="brief", n_names=5, tol=0.10):
    """Sanity checks on the spread ladder.
    
    Three tests that every sensible basket should pass:
    1. Monotonicity: s_1 >= s_2 >= ... >= s_n (first-to-default is the riskiest)
    2. Bounds: first-to-default spread is between the max single-name spread
       (scaled) and the sum of all single-name spreads (scaled)
    3. Sum identity: sum of basket spreads approximately equals sum of single
       spreads (scaled) - this one is approximate due to annuity mismatches
    
    The sum identity is the weakest test. Due to different annuity discounting
    across the k's, it can be off by 5-8% even when everything is correct.
    """
    ks = sorted(res)
    s = np.array([res[k]["spread_bps"] for k in ks])
    single = np.asarray(single_spreads, float)
    scale = (1.0 / n_names) if payoff == "brief" else 1.0

    checks = {}
    checks["monotonic"] = {
        "pass": bool(np.all(np.diff(s) <= 1e-6)),
        "detail": " >= ".join(f"{v:.2f}" for v in s),
    }
    lo, hi = scale * single.max(), scale * single.sum()
    checks["bounds"] = {
        "pass": bool(lo - 1e-6 <= s[0] <= hi + 1e-6),
        "detail": f"{lo:.2f} <= s_1 = {s[0]:.2f} <= {hi:.2f} bps",
    }
    lhs, rhs = s.sum(), scale * single.sum()
    checks["sum_identity (approx)"] = {
        "pass": bool(abs(lhs - rhs) / rhs < tol),
        "detail": f"sum_k s_k = {lhs:.2f} vs scale*sum_i s_i = {rhs:.2f} bps "
                  f"({100*(lhs-rhs)/rhs:+.1f}%, annuity mismatch)",
    }
    return checks
