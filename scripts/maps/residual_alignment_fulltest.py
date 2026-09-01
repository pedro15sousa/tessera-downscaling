"""Full-test-set residual-alignment of the TESSERA increment, all regions.

Generalises the dense-maps paragraph-2 diagnostic from the mapped snapshots to the
ENTIRE test set of every region, using the MAIN-RESULTS models (Gaussian+mTPI t2m,
truncated-normal+mTPI wind) straight from each run's saved test_predictions.npz --
no re-inference. Per (station, obs) pair:

    e = y - yhat_base      (signed baseline error)
    D = yhat_tess - yhat_base   (signed TESSERA increment)
    g = |e| - |e - D|      (per-obs MAE gain)

Point estimate matches cross_folder_analysis (MAE@median): Gaussian median==mu;
truncated-normal median via the head's numerically-stable quantile
    x = mu - sigma * Phi^-1( Phi(mu/sigma) * (1-p) ),  p=0.5
(replicated here in float64 with scipy.special.ndtri, floored at 0 -- the naive
mu + sigma*Phi^-1(0.5(1+Phi(-mu/sigma))) overflows to +-inf for mu<<0, e.g. us wind).

Reports, per variable per region: R^2 = corr(D,e)^2 (baseline-error variance the
increment explains), directional hit-rate, rho(|D|,g) vs a random-increment null,
and -- for context -- the aggregate CRPS of both models (seed-mean, from
test_summary.json). CRPS is a distributional proper score and does NOT enter the
regression, which is intrinsically about the signed POINT prediction.

Imported by notebooks/residual_structure_analysis.ipynb (§4) and runnable on its
own:  uv run python scripts/maps/residual_alignment_fulltest.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy.special import log_ndtr, ndtri
from scipy.stats import pearsonr, spearmanr

from tessera_downscaling.paths import training_runs_dir

SEEDS = [42, 123, 456]
REGIONS = ["eu", "us", "east_asia", "australia", "southern_africa"]
STEM = {
    "t2m": (
        "t2m_snap_bilinear_baseline_mtpi_wd",
        "t2m_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd",
    ),
    "wind": (
        "wind_truncnormal_snap_bilinear_baseline_mtpi_wd",
        "wind_truncnormal_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd",
    ),
}


def tn_median(mu, sigma):
    """Stable truncated-normal (lower bound 0) median; matches TruncatedNormalHead."""
    Z = np.exp(log_ndtr(mu / sigma))  # Phi(mu/sigma)
    q = np.clip(Z * 0.5, 1e-300, 1.0 - 1e-15)
    return np.clip(mu - sigma * ndtri(q), 0.0, None)


def point(d, var):
    mu = d[f"{var}_param_mu"].astype(np.float64)
    if var == "t2m":
        return mu  # gaussian median == mu
    sigma = np.exp(0.5 * d[f"{var}_param_log_var"].astype(np.float64))
    return tn_median(mu, sigma)


def ens_point(reg, stem, var, suffix=""):
    """suffix selects a latents-generation retrain folder, e.g.
    "_tessera_1B-M_2017" -> training_runs/snapshot_14y_<reg>_tessera_1B-M_2017."""
    pts, tgt = [], None
    for s in SEEDS:
        f = (
            training_runs_dir(f"snapshot_14y_{reg}{suffix}")
            / f"{stem}_seed{s}"
            / "test_predictions.npz"
        )
        if not f.exists():
            return None
        d = np.load(f)
        pts.append(point(d, var))
        tgt = d[f"{var}_targets"].astype(np.float64)
    return np.mean(pts, 0), tgt


def crps_seedmean(reg, stem, var, suffix=""):
    vals = []
    for s in SEEDS:
        f = (
            training_runs_dir(f"snapshot_14y_{reg}{suffix}")
            / f"{stem}_seed{s}"
            / "test_summary.json"
        )
        if f.exists():
            v = json.load(open(f)).get(f"{var}_crps")
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    rng = np.random.default_rng(0)
    print(
        "Full test set, MAIN-RESULTS models (t2m gaussian+mTPI, wind truncnormal+mTPI),"
    )
    print("3-seed, MAE@median point estimate (stable truncated median for wind):\n")
    hdr = (
        "region",
        "var",
        "n",
        "MAEb",
        "MAEt",
        "dMAE%",
        "CRPSb",
        "CRPSt",
        "dCRPS%",
        "R2",
        "hit%",
        "rho",
        "null",
    )
    print(
        "".join(
            f"{h:>{w}}"
            for h, w in zip(hdr, [16, 5, 9, 7, 7, 6, 7, 7, 7, 6, 6, 6, 7], strict=False)
        )
    )
    pool = {}
    for reg in REGIONS:
        for var in ["t2m", "wind"]:
            rb = ens_point(reg, STEM[var][0], var)
            rt = ens_point(reg, STEM[var][1], var)
            if rb is None or rt is None:
                print(f"{reg:>16}{var:>5}  (missing)")
                continue
            yb, tb = rb
            yt, tt = rt
            if not (len(tb) == len(tt) and np.allclose(tb, tt)):
                print(f"{reg:>16}{var:>5}  [misaligned] nb={len(tb)} nt={len(tt)}")
                continue
            y = tb
            e = y - yb
            d = yt - yb
            g = np.abs(e) - np.abs(e - d)
            r = pearsonr(d, e).statistic
            hit = np.mean(np.sign(d) == np.sign(e))
            idx = rng.choice(len(e), size=min(80000, len(e)), replace=False)
            es, ds = e[idx], d[idx]
            rho = spearmanr(np.abs(ds), g[idx]).statistic
            null = np.mean(
                [
                    spearmanr(
                        np.abs(f := rng.normal(0, ds.std(), len(ds))),
                        np.abs(es) - np.abs(es - f),
                    ).statistic
                    for _ in range(20)
                ]
            )
            maeb, maet = np.abs(e).mean(), np.abs(e - d).mean()
            cb = crps_seedmean(reg, STEM[var][0], var)
            ct = crps_seedmean(reg, STEM[var][1], var)
            dmae = 100 * (maeb - maet) / maeb
            dcrps = 100 * (cb - ct) / cb if cb == cb else float("nan")
            print(
                f"{reg:>16}{var:>5}{len(e):>9}{maeb:>7.2f}{maet:>7.2f}{dmae:>6.1f}"
                f"{cb:>7.2f}{ct:>7.2f}{dcrps:>7.1f}{r**2:>6.2f}{100 * hit:>6.1f}"
                f"{rho:>6.2f}{null:>7.2f}"
            )
            pool.setdefault(var, []).append((e, d))
    print("\n=== pooled across aligned regions ===")
    for var in ["t2m", "wind"]:
        E = np.concatenate([x[0] for x in pool[var]])
        D = np.concatenate([x[1] for x in pool[var]])
        r = pearsonr(D, E).statistic
        hit = np.mean(np.sign(D) == np.sign(E))
        print(
            f"  {var}: N={len(E):,}  R2={r**2:.2f}  pearson={r:+.2f}  hit={100 * hit:.1f}%"
        )


if __name__ == "__main__":
    main()
