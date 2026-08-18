"""Norway temporal-rollout — latent-space corroboration of the flat held-out curve.

Why does the held-out (Europe ``spatial_test``) curve in
``snapshot_14y_eu_temporal_rollout_norway`` barely move while the Norway
``probe`` curve improves a lot? This script corroborates the paper's claim in
the TESSERA latent space the model actually conditions on
(``station_latents_lat16_grad0.5.npy``, 16-d; projected to 8-d at train time):

  Q1  Held-out test stations are WITHIN the training distribution -> downscaling
      reduces to interpolation -> they sit near a ceiling the rollout can't push.
  Q2  Norway probe/train stations are a distinct, separable sub-population
      (OOD-ish) -> the data rolled in is Norway-specific.
  Link Norway training stations contribute ~0 to the nearest-neighbour coverage
      of the (85%) non-Norway held-out stations -> rolling Norway in cannot move
      the aggregate held-out curve; it only helps the ~15% Norway held-out
      stations. Confirmed directly in the per-station model errors.

Run:
    .venv/bin/python projects/tessera_downscaling/scripts/norway_rollout_descriptors/norway_ood_latent_analysis.py
Outputs (figures + report.json):
    projects/tessera_downscaling/notebooks/norway_analysis_outputs/
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "projects/tessera_downscaling/.tmp_output"
OUT = REPO / "projects/tessera_downscaling/notebooks/norway_analysis_outputs"
OUT.mkdir(exist_ok=True, parents=True)

ROLLOUT = "training_runs_snapshot_14y_eu_temporal_rollout_norway"
SWEEPS = ["r1mo", "r3mo", "r6mo", "r1y", "r2y", "r3y", "r6y"]
SWEEP_YEARS = {"r1mo": 1/12, "r3mo": .25, "r6mo": .5, "r1y": 1., "r2y": 2., "r3y": 3., "r6y": 6.}
SEEDS = [42, 123, 456]
NB_LAT, NB_LON = (58.0, 71.0), (4.0, 31.0)
TESSERA = "t2m_snap_vae_lat16_proj8_concat_with_elev_no_static_wd"
BASELINE = "t2m_snap_bilinear_baseline_wd"

report: dict = {}

# ==========================================================================
# Load latents + station table, build groups
# ==========================================================================
# ---- TESSERA latents generation (pick ONE) -----------------------------
# Current main: TESSERA v2 "1B-M", 2017 embeddings (crop64_lat16_auxon).
LATENTS_NPY = BASE / ("processed/vae_tessera_1B-M/"
                      "station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy")
# Previous main: TESSERA v1 16-d latents.
# LATENTS_NPY = BASE / "processed/station_latents_lat16_grad0.5.npy"

lat = np.load(LATENTS_NPY)
ll = pd.read_csv(BASE / "processed/tessera_global/station_list_filtered.csv")
ll["station_id"] = ll["station_id"].astype(str)
lat_of = {sid: i for i, sid in enumerate(ll["station_id"])}

st = pd.read_csv(BASE / "dataset_timestamp_global/stations.csv")
st["station_id"] = st["station_id"].astype(str)
st["lrow"] = st["station_id"].map(lat_of)
st = st[st["lrow"].notna()].copy(); st["lrow"] = st["lrow"].astype(int)
st = st[~np.isnan(lat[st["lrow"].to_numpy()]).any(axis=1)].reset_index(drop=True)
Z16 = lat[st["lrow"].to_numpy()]

in_nb = (st.latitude.between(*NB_LAT) & st.longitude.between(*NB_LON) & (st.region == "europe")).to_numpy()
is_eu, is_tr, is_te = (st.region == "europe").to_numpy(), (st.spatial_split == "train").to_numpy(), (st.spatial_split == "test").to_numpy()
g = {
    "norway_probe":  in_nb & is_eu & is_tr,
    "norway_test":   in_nb & is_eu & is_te,
    "eu_train_rest": is_eu & is_tr & ~in_nb,
    "eu_test_rest":  is_eu & is_te & ~in_nb,
    "eu_train_full": is_eu & is_tr,
    "eu_test_full":  is_eu & is_te,
}
report["group_sizes"] = {k: int(v.sum()) for k, v in g.items()}

scaler = StandardScaler().fit(Z16[g["eu_train_full"]])
Zs = scaler.transform(Z16)
def sub(m): return Zs[m]

# ==========================================================================
# Q2a separability (Norway-train vs rest-EU-train)
# ==========================================================================
Xa = np.vstack([sub(g["norway_probe"]), sub(g["eu_train_rest"])])
ya = np.r_[np.ones(g["norway_probe"].sum()), np.zeros(g["eu_train_rest"].sum())]
auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                      Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
report["Q2a_separability"] = {"roc_auc_mean": float(auc.mean()), "roc_auc_std": float(auc.std()),
                              "norway_base_rate": float(g["norway_probe"].sum() / g["eu_train_full"].sum())}

# ==========================================================================
# Q2b kNN purity within EU train
# ==========================================================================
K = 10
etZ, etN = sub(g["eu_train_full"]), in_nb[g["eu_train_full"]]
_, idx = NearestNeighbors(n_neighbors=K + 1).fit(etZ).kneighbors(etZ)
nb_isnor = etN[idx[:, 1:]]
report["Q2b_knn_purity_k10"] = {
    "norway_nbrs_that_are_norway": float(nb_isnor[etN].mean()),
    "rest_nbrs_that_are_rest": float((~nb_isnor[~etN]).mean()),
    "norway_base_rate": float(etN.mean())}

# ==========================================================================
# Q2c Mahalanobis to non-Norway EU train
# ==========================================================================
cov = EmpiricalCovariance().fit(sub(g["eu_train_rest"]))
mh = {k: np.sqrt(cov.mahalanobis(sub(v))) for k, v in g.items()}
y_ood = np.r_[np.ones(g["norway_probe"].sum()), np.zeros(g["eu_test_rest"].sum())]
s_ood = np.r_[mh["norway_probe"], mh["eu_test_rest"]]
report["Q2c_mahalanobis_to_rest"] = {"median": {k: float(np.median(v)) for k, v in mh.items()},
                                     "ood_auc_norway_vs_nonnorway_test": float(roc_auc_score(y_ood, s_ood))}

# ==========================================================================
# Q2d region centroid distances from Norway
# ==========================================================================
cents = {}
for r in st.region.unique():
    m = ((st.region == r) & is_tr).to_numpy()
    if m.sum(): cents[r] = Zs[m].mean(0)
cents["europe_rest"] = Zs[g["eu_train_rest"]].mean(0); cents["norway"] = Zs[g["norway_probe"]].mean(0)
report["Q2d_region_dist_from_norway"] = {r: float(np.linalg.norm(cents["norway"] - cents[r])) for r in cents if r != "norway"}
report["Q2d_region_dist_from_norway"]["norway_internal_spread"] = float(np.mean(np.linalg.norm(sub(g["norway_probe"]) - cents["norway"], axis=1)))

# ==========================================================================
# Q1 interpolation test + Link (coverage decomposition)
# ==========================================================================
def nn_to(qm, rm, loo=False):
    k = 2 if loo else 1
    d, _ = NearestNeighbors(n_neighbors=k).fit(sub(rm)).kneighbors(sub(qm))
    return d[:, -1]
d_loo = nn_to(g["eu_train_full"], g["eu_train_full"], loo=True)
q95 = float(np.percentile(d_loo, 95))
queries = {"eu_test_rest": g["eu_test_rest"], "norway_test": g["norway_test"], "eu_test_full": g["eu_test_full"]}
nn_full = {k: nn_to(m, g["eu_train_full"]) for k, m in queries.items()}
nn_rest = {k: nn_to(m, g["eu_train_rest"]) for k, m in queries.items()}
report["Q1_interpolation"] = {"train_loo_median": float(np.median(d_loo)), "train_loo_q95": q95,
    "nn_full": {k: {"median": float(np.median(v)), "frac_within_train_q95": float(np.mean(v <= q95))} for k, v in nn_full.items()}}
etfZ, etfN = sub(g["eu_train_full"]), in_nb[g["eu_train_full"]]
nbr1 = NearestNeighbors(n_neighbors=1).fit(etfZ)
link = {}
for k, m in queries.items():
    _, ii = nbr1.kneighbors(sub(m))
    link[k] = {"median_nn_full": float(np.median(nn_full[k])),
               "median_nn_rest_excl_norway": float(np.median(nn_rest[k])),
               "median_delta_from_norway": float(np.median(nn_rest[k] - nn_full[k])),
               "frac_nearest_train_is_norway": float(etfN[ii[:, 0]].mean())}
report["Link_norway_coverage"] = link

# ==========================================================================
# Rollout curves split: held-out Norway vs non-Norway (per-station model errors)
# ==========================================================================
VAR_BASE = {  # (variable) -> {family: run-dir prefix}
    "t2m":  {"tessera": TESSERA, "baseline": BASELINE},
    "wind": {"tessera": "wind_snap_vae_lat16_proj8_concat_with_elev_no_static_wd",
             "baseline": "wind_snap_bilinear_baseline_wd"},
}

def split_curve(base, var):
    out = {"norway": [], "nonnorway": []}
    for sw in SWEEPS:
        acc = {}
        for seed in SEEDS:
            f = BASE / ROLLOUT / f"{base}_{sw}_seed{seed}" / "test_station_errors.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            sid, mae, cnt = d["station_ids"], d[f"{var}_station_mae"], d[f"{var}_station_count"]
            la, lo, ss = d["station_lats"], d["station_lons"], d["subset_per_station"]
            for i in range(len(sid)):
                if cnt[i] <= 0 or ss[i] != "spatial_test":
                    continue
                a = acc.setdefault(sid[i], [0., 0, la[i], lo[i]])
                a[0] += mae[i] * cnt[i]; a[1] += cnt[i]
        nor, rest = [], []
        for _, (se, c, la, lo) in acc.items():
            isn = (NB_LAT[0] <= la <= NB_LAT[1]) and (NB_LON[0] <= lo <= NB_LON[1])
            (nor if isn else rest).append(se / c)
        out["norway"].append(float(np.mean(nor))); out["nonnorway"].append(float(np.mean(rest)))
    return out

curves = {var: {fam: split_curve(base, var) for fam, base in fams.items()}
          for var, fams in VAR_BASE.items()}
report["rollout_heldout_split_macro_mae"] = {
    var: {fam: {"norway": [round(x, 4) for x in c["norway"]],
                "nonnorway": [round(x, 4) for x in c["nonnorway"]],
                "delta_norway": round(c["norway"][-1] - c["norway"][0], 4),
                "delta_nonnorway": round(c["nonnorway"][-1] - c["nonnorway"][0], 4)}
          for fam, c in fcurves.items()}
    for var, fcurves in curves.items()}

# ==========================================================================
# FIGURES
# ==========================================================================
pca = PCA(2, random_state=0).fit(sub(g["eu_train_full"]))
P = {k: pca.transform(sub(v)) for k, v in g.items()}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].scatter(*P["eu_train_rest"].T, s=4, c="#cccccc", label=f"EU train, non-Norway (n={g['eu_train_rest'].sum()})")
ax[0].scatter(*P["norway_probe"].T, s=6, c="#1f77b4", label=f"Norway probe/train (n={g['norway_probe'].sum()})")
ax[0].set_title("(a) Norway is a distinct sub-cluster in TESSERA latent space\n"
                f"linear AUC={report['Q2a_separability']['roc_auc_mean']:.3f}, "
                f"kNN-purity={report['Q2b_knn_purity_k10']['norway_nbrs_that_are_norway']:.2f} (base {report['Q2b_knn_purity_k10']['norway_base_rate']:.2f})")
ax[1].scatter(*P["eu_train_rest"].T, s=3, c="#eeeeee", label="EU train (context)")
ax[1].scatter(*P["norway_probe"].T, s=5, c="#9ecae1", label="Norway train")
ax[1].scatter(*P["norway_test"].T, s=24, c="#d62728", marker="x", label=f"Norway test (n={g['norway_test'].sum()})")
ax[1].set_title("(b) Norway held-out test lies INSIDE the Norway train cluster\n(interpolable, but only from Norway train)")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig1_pca_norway_cluster.png", dpi=140); plt.close(fig)

def ecdf(ax, d, label, **kw):
    x = np.sort(d); ax.plot(x, np.arange(1, len(x)+1)/len(x), label=f"{label} (med={np.median(d):.2f})", **kw)
fig, ax = plt.subplots(figsize=(8, 5))
ecdf(ax, d_loo, "EU train↔train (in-dist ref)", c="k", lw=2)
ecdf(ax, nn_full["eu_test_rest"], "non-Norway test → EU train", c="#2ca02c")
ecdf(ax, nn_full["norway_test"], "Norway test → FULL EU train", c="#1f77b4")
ecdf(ax, nn_rest["norway_test"], "Norway test → EU train EXCL Norway", c="#d62728", ls="--")
ax.axvline(q95, c="grey", ls=":", label=f"train q95={q95:.2f}")
ax.set_xlabel("nearest-neighbour distance (standardized lat16)"); ax.set_ylabel("ECDF")
ax.set_title("Held-out = interpolation; Norway test interpolable only from Norway train")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig2_nn_interpolation.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, np.percentile(mh["norway_probe"], 99), 60)
for k, c in [("eu_test_rest", "#2ca02c"), ("eu_train_rest", "#999999"), ("norway_probe", "#1f77b4"), ("norway_test", "#d62728")]:
    ax.hist(mh[k], bins=bins, density=True, histtype="step", lw=2, color=c, label=f"{k} (med={np.median(mh[k]):.1f})")
ax.set_xlabel("Mahalanobis dist to non-Norway EU train"); ax.set_ylabel("density")
ax.set_title(f"OOD shift: Norway vs the rest of EU train (OOD AUC={report['Q2c_mahalanobis_to_rest']['ood_auc_norway_vs_nonnorway_test']:.3f})")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig3_mahalanobis_ood.png", dpi=140); plt.close(fig)

x = [SWEEP_YEARS[s] for s in SWEEPS]
sty = {"tessera": "-", "baseline": "--"}
units = {"t2m": "°C", "wind": "m/s"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
for ax, var in zip(axes, ["t2m", "wind"]):
    for fam in ["tessera", "baseline"]:
        c = curves[var][fam]
        ax.plot(x, c["norway"], sty[fam], color="#d62728", marker="o", ms=5,
                label=f"{fam}: Norway held-out (14%)")
        ax.plot(x, c["nonnorway"], sty[fam], color="#2ca02c", marker="s", ms=5,
                label=f"{fam}: non-Norway held-out (86%)")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(SWEEPS, fontsize=8)
    ax.set_xlabel("elapsed since rollout start")
    ax.set_ylabel(f"{var} held-out MAE ({units[var]}, macro)")
    dn = report["rollout_heldout_split_macro_mae"][var]["baseline"]
    ax.set_title(f"{var}: held-out gain is Norway-only "
                 f"(baseline non-Norway Δ={dn['delta_nonnorway']:+.2f})")
    ax.grid(alpha=.3); ax.legend(fontsize=7.5)
fig.suptitle("Held-out improvement over the Norway rollout is concentrated in the "
             "Norwegian held-out stations; the non-Norway majority is ~flat", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "fig5_heldout_split_rollout.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(st.longitude[g["eu_train_rest"]], st.latitude[g["eu_train_rest"]], s=2, c="#dddddd", label="EU train (rest)")
ax.scatter(st.longitude[g["norway_probe"]], st.latitude[g["norway_probe"]], s=6, c="#1f77b4", label="Norway probe")
ax.scatter(st.longitude[g["norway_test"]], st.latitude[g["norway_test"]], s=18, c="#d62728", marker="x", label="Norway test")
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title("Geographic footprint"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig4_geography.png", dpi=140); plt.close(fig)

(OUT / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("\nWritten to", OUT)
