# k-th-to-Default Basket CDS Pricing (with an IRRBB / CSRBB extension)

**CQF Final Project, January 2026 cohort. Project grade: 92%.**

This project prices the full k-th-to-default ladder (k = 1 to 5) on a five-name basket credit default swap under Gaussian and Student-t copulas, using simulation in Python. It then applies an ALM view to the same position by separating interest-rate risk from credit-spread risk under the BCBS 368 supervisory shocks.

---

## In plain terms

A basket CDS covers several companies in one contract. A k-th-to-default basket pays out only when the k-th company in the basket defaults: the 1st-to-default pays on the first default, the 2nd-to-default on the second, and so on up to the 5th.

Pricing it needs two things:

1. **How likely each company is to default.** This comes from its CDS spreads.
2. **How likely the companies are to default together.** This comes from a copula fitted to equity returns.

There is no closed-form answer for the k-th default time, so I simulate correlated default times for all five names over many scenarios and average the cash flows.

---

## Contract and data

| Item | Value |
|---|---|
| Reference names | Volkswagen, Deutsche Bank, TotalEnergies, Siemens, Telefónica |
| Notional | EUR 2m per name, EUR 10m in total |
| Maturity / premium | 5 years, paid quarterly in arrears |
| Recovery | 40% (Siemens 45%) |
| CDS curves (1Y to 5Y) | **Assumed** upward-sloping term structure, scaled to a 5Y anchor per name (40 to 120 bp). These are not market quotes. |
| Dependence data | Weekly equity log-returns, Aug 2023 to Aug 2026 (156 observations), used as a proxy for default dependence |
| Discount curve | Nelson–Siegel, 2.8% long-run level |

---

## Method

1. **Marginals.** I bootstrap piecewise-constant hazard rates pillar by pillar from each 1Y to 5Y CDS strip. The stripped curves reprice their input spreads with an error of 0.0 bp.
2. **Dependence.** Equity returns are converted to pseudo-uniforms with a kernel-smoothed CDF and GPD tails. Correlation comes from rank statistics, and the Student-t degrees of freedom are estimated by explicit maximum likelihood (ν̂ = 17.86).
3. **Simulation.** I draw correlated uniforms with **scrambled Sobol** sequences (2¹⁶ = 65,536 paths). For the t-copula, the χ² draw takes its own Sobol dimension so the low-discrepancy property is kept. Each name's survival curve is inverted to get exact default times, which are then sorted.
4. **Pricing.** The protection and premium legs are averaged across paths separately, and the ratio is taken once. Taking the ratio path by path would bias the result.
5. **Validation.** Four checks: monotonicity, no-arbitrage bounds, an approximate spread-sum check, and an **exact protection-leg identity**.
6. **Risk.** Sensitivities to correlation, single-name spreads and recovery, followed by an IRRBB vs CSRBB decomposition.

---

## Mathematical framework

### 1. Hazard rates and survival

Each name's default time $`\tau`$ is the first jump of a Poisson process with piecewise-constant intensity $`\lambda_j`$ on each yearly interval $`\Delta t_j`$:

```math
Q(t) \;=\; \Pr(\tau > t) \;=\; \exp\!\Big(-\int_0^{t}\lambda_u\,du\Big) \;=\; \exp\!\Big(-\sum_{j\le m}\lambda_j\,\Delta t_j\Big)
```

For a single-name CDS with spread $`s`$, recovery $`R`$ and risk-free discount factor $`Z(0,t)`$, the par condition sets the two legs equal:

```math
\underbrace{(1-R)\sum_{i} Z(0,t_i)\,\big[Q(t_{i-1})-Q(t_i)\big]}_{\text{protection leg}}
\;=\;
\underbrace{s\sum_{i} Z(0,t_i)\,Q(t_i)\,\Delta t_i}_{\text{premium leg}} \;+\; \text{accrual}
```

Bootstrapping solves this one maturity at a time. Once the survival probabilities are known, each hazard rate follows:

```math
\lambda_m \;=\; -\frac{1}{\Delta t_m}\,\ln\frac{Q(t_m)}{Q(t_{m-1})}
```

### 2. Simulating default times

With cumulative hazard $`\Lambda(t)=\sum_j \lambda_j\,\Delta t_j`$, a uniform draw $`u`$ maps to a default time through $`Q(\tau)=u`$:

```math
\Lambda(\tau) = -\ln u,
\qquad
\tau = t_{m-1} + \frac{-\ln u \;-\; \Lambda(t_{m-1})}{\lambda_m}
```

where $`m`$ is the first year in which $`\Lambda(t_m) > -\ln u`$.

### 3. Copulas

Rank correlations are converted to the linear correlation the copula needs:

```math
\rho = \sin\!\Big(\frac{\pi}{2}\,\tau_K\Big) \quad \text{(Kendall)},
\qquad
\rho = 2\sin\!\Big(\frac{\pi}{6}\,\rho_S\Big) \quad \text{(Spearman)}
```

**Gaussian copula.** With $`\Sigma = AA^{\top}`$ (Cholesky factor) and $`\mathbf{z}\sim N(\mathbf{0},\mathbf{I})`$:

```math
\mathbf{x} = A\mathbf{z},
\qquad
\mathbf{u} = \Phi(\mathbf{x}),
\qquad
C^{Ga}_{\Sigma}(\mathbf{u}) = \Phi_{\Sigma}\big(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_n)\big)
```

**Student-t copula.** All names are divided by one shared $`\chi^2`$ draw, so extreme moves happen together:

```math
s \sim \chi^2_{\nu},
\qquad
\mathbf{y} = \frac{A\mathbf{z}}{\sqrt{s/\nu}},
\qquad
\mathbf{u} = T_{\nu}(\mathbf{y})
```

This gives non-zero tail dependence. The Gaussian copula has none for any $`\rho<1`$:

```math
\lambda_L = \lambda_U = 2\,t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right) > 0
```

The degrees of freedom are estimated by maximising the copula log-likelihood over the pseudo-samples $`\hat{\mathbf{u}}_t`$:

```math
\hat{\nu} = \arg\max_{\nu}\;\sum_{t=1}^{T}\ln c^{t}_{\Sigma,\nu}\big(\hat{\mathbf{u}}_t\big)
```

### 4. k-th-to-default pricing

On each path the simulated default times are sorted:

```math
\tau_{(1)} \le \tau_{(2)} \le \dots \le \tau_{(k)} \le \dots \le \tau_{(n)}
```

The seller pays $`(1-R)\,\tfrac{1}{n}\,N`$ at $`\tau_{(k)}`$ if $`\tau_{(k)} \le T`$, and premiums run until $`\min(\tau_{(k)},T)`$. With $`F_k`$ the distribution of $`\tau_{(k)}`$, the fair spread is:

```math
s_k \;=\; \frac{\mathbb{E}[\text{DL}_k]}{\mathbb{E}[\text{PL}_k]}
\;=\;
\frac{(1-R)\sum_{i} Z(0,t_i)\,\big[F_k(t_i)-F_k(t_{i-1})\big]}
{\sum_{i} Z(0,t_i)\,\big[1-F_k(t_i)\big]\,\Delta t_i}
```

In the simulation the two legs are averaged across all $`M`$ paths first, and the ratio is taken once:

```math
\hat{s}_k \;=\; \frac{\tfrac{1}{M}\sum_{p=1}^{M}\text{DL}_k^{(p)}}{\tfrac{1}{M}\sum_{p=1}^{M}\text{PL}_k^{(p)}}
```

Higher-seniority baskets are always cheaper:

```math
s_1 \ge s_2 \ge \dots \ge s_n
```

### 5. Exact validation identity

Every default before $`T`$ is the k-th default for exactly one $`k`$, so the protection legs of the five baskets add up to the protection legs of the five single names:

```math
\sum_{k=1}^{n}\mathbb{E}\big[\text{DL}_k\big] \;=\; \sum_{i=1}^{n}\mathbb{E}\big[\text{DL}_i\big]
```

No annuity appears in this identity, so it tests the sorting, recovery and discounting directly. It holds to a relative error of 0.0.

### 6. Simulation error

```math
\text{Monte Carlo: } \;\varepsilon \sim O\big(N^{-1/2}\big),
\qquad
\text{Quasi-Monte Carlo (Sobol): } \;\varepsilon \sim O\!\left(\frac{(\ln N)^{d}}{N}\right)
```

### 7. IRRBB shocks (BCBS 368, EUR calibration)

With $`\Delta R_{\text{parallel}} = 200`$ bp, $`\Delta R_{\text{short}} = 250`$ bp, $`\Delta R_{\text{long}} = 100`$ bp and $`x = 4`$:

```math
S_{\text{short}}(t) = \Delta R_{\text{short}}\,e^{-t/x},
\qquad
S_{\text{long}}(t) = \Delta R_{\text{long}}\,\big(1-e^{-t/x}\big)
```

```math
S_{\text{steepener}}(t) = -0.65\,\big|S_{\text{short}}(t)\big| + 0.9\,\big|S_{\text{long}}(t)\big|,
\qquad
S_{\text{flattener}}(t) = 0.8\,\big|S_{\text{short}}(t)\big| - 0.6\,\big|S_{\text{long}}(t)\big|
```

The change in economic value under scenario $`j`$, and the supervisory outlier test:

```math
\Delta \text{EVE}_j = \text{EVE}_j - \text{EVE}_0,
\qquad
\text{SOT breach if } \;\frac{\max_j\big(-\Delta\text{EVE}_j\big)}{\text{Tier 1}} > 15\%
```

---

## Key results

**Fair spreads (bp, brief notional convention, 65,536 scrambled Sobol paths)**

| | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Gaussian copula | 65.10 | 13.61 | 2.88 | 0.37 | 0.022 |
| Student-t copula (ν = 17.86) | 63.01 | 14.48 | 3.41 | 0.57 | 0.047 |

- **Validation.** Both ladders are monotone, and the 1st-to-default lies inside its bounds of [24.00, 76.00] bp. The exact protection-leg identity holds to a relative error of **0.0** under both copulas.
- **Variance reduction.** Scrambled Sobol cuts the standard error on the 1st-to-default by **6.9x** compared with plain Monte Carlo (12 independent scrambles). The gain narrows to about 2x for the most senior basket.
- **Correlation moves risk between seniorities.** Moving to ρ = 0.90 takes the 1st-to-default from 65.10 to 31.74 bp and the 5th-to-default from 0.022 to 4.82 bp. Total expected loss is set by the marginals; correlation decides which layer absorbs it.
- **Recovery sensitivity reverses once the curves are re-stripped.** Holding hazards fixed makes spreads scale exactly with (1 − R), which is an identity, not a sensitivity. Re-bootstrapping at each R gives a much smaller, non-linear effect, and senior baskets become *more* expensive as R rises.

---

## Extension: IRRBB vs CSRBB

The risk-free curve (Nelson–Siegel) and the credit curves (from the CDS strip) are separate risk factors in the code, so I shock each one on its own and measure the change in economic value of a protection-seller book that holds one unit of each basket.

- **IRRBB.** The six BCBS 368 shocks on the EUR calibration (parallel ±200 bp, short ±250 bp, steepener and flattener). The worst ΔEVE is EUR −1,515 (parallel down), which is well inside the 15%-of-Tier-1 supervisory outlier test against an *assumed* Tier 1 of EUR 50m.
- **CSRBB.** Credit spreads shocked by +50, +100 and +200 bp. The worst ΔEVE is EUR −904,855. This is reported for monitoring only, because EBA/GL/2022/14 sets no numeric outlier threshold for CSRBB.
- **What the split shows.** Credit-spread risk is about 600 times the interest-rate risk on this book. That is expected: the seller is short protection on five names, and the risk-free curve only enters through discounting.

> **Scope note.** A basket CDS is a trading-book product. This section is a methodological analogue: it shows how the same machinery separates rate risk from spread risk, as an ALM team would. It does not claim that the instrument falls inside the banking-book CSRBB perimeter.

---

## Repository contents

| File | Purpose |
|---|---|
| `FinalProject_CR_BHUPENDRA_RAMTEKE.ipynb` | Main notebook: theory, calibration, pricing, validation, sensitivities, extension |
| `basket_engine.py` | Companion module: hazard bootstrapping, copulas, t-copula MLE, pricing, validation checks |
| `download_prices.py` | One-off script that downloads the equity price history |
| `equity_prices_daily.csv` | Daily closes used to build the weekly returns |
| `requirements.txt` | Python dependencies |

## How to run

```bash
git clone https://github.com/bhupendraquants123/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook FinalProject_CR_BHUPENDRA_RAMTEKE.ipynb
```

The notebook reads `equity_prices_daily.csv` directly, so it does not need internet access. `basket_engine.py` must be in the same folder.

---

## Limitations

- The CDS curves are **assumed**, and every name uses the same curve shape, scaled to its own 5Y level.
- Equity returns are only a **proxy** for default dependence. The average correlation is low (0.21) because TotalEnergies moved fairly independently of the other names in this window.
- ν̂ sits on a flat likelihood. The likelihood ratio against a near-Gaussian copula is only 2.4 on 156 observations, and the window has no major stress event.
- GPD tails are fitted on about 16 points each.
- The model is static, with constant intensities, recoveries and correlation, and has no counterparty or CVA risk.
- The 5th-to-default triggers on about one path in 12,945, so its price carries noticeable simulation noise.

## Possible extensions

- Importance sampling for the senior baskets
- A Clayton copula to capture asymmetric (lower-tail) dependence
- An independent semi-analytic benchmark, such as Hull–White
- Stochastic recovery correlated with the systematic factor, and CVA on the protection seller
- Generalising to CDO tranches calibrated to iTraxx base correlations

---

## References

- CQF Institute lectures and final project workshops, January 2026 cohort (intensity models, CDS pricing, CDO and correlation sensitivity)
- MathWorks, *Using Extreme Value Theory and Copulas to Evaluate Market Risk*
- BCBS, *Interest Rate Risk in the Banking Book* (d368), April 2016
- EBA, *Guidelines on IRRBB and CSRBB* (EBA/GL/2022/14), 2022

---

*Author: Bhupendra Ramteke, CFA, FRM, CQF*
