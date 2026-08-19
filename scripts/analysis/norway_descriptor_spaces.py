"""Why TESSERA transfers to Norway: reachability vs faithfulness across descriptor spaces.

Question: when predicting a Norwegian station, can the model find genuinely
similar surfaces among its (non-Norwegian) training stations? That depends on
the SPACE in which "similar" is measured. We compare three per-station
descriptor spaces:

  1. geographic     — (latitude, longitude)
  2. elevation+mTPI — (elevation, delta_elevation, mTPI)   [the baseline's
                       per-station inputs, minus the coarse ERA5 static grids]
  3. TESSERA lat16  — the 16-d land-surface embedding the model conditions on

For each space we ask, for the held-out Norwegian stations against the
NON-Norwegian European training stations:

  * Isolation factor  — median NN(Norway -> rest) / median NN(rest -> rest, LOO).
                        Dimensionless; =1 means Norway sits as close to the
                        training set as training points sit to one another;
                        >1 means more isolated. (A ratio, so it is comparable
                        ACROSS spaces of different dimensionality — raw
                        distances are not.)
  * Reachability      — fraction of Norwegian stations whose nearest
                        non-Norwegian training neighbour lies within R95, the
                        95th-percentile of the rest->rest LOO NN distances
                        (i.e. has an "in-distribution" European analogue).
  * Faithfulness      — separability AUC: 4-fold CV ROC-AUC of L2-regularised
    (separability AUC)  logistic regression classifying Norway vs rest-EU-train
                        in the standardised space. 0.5 = indistinguishable;
                        1.0 = the descriptor fully separates Norway.

Each space is z-scored on the rest-EU-train reference before any distance is
taken, so within-space distances are scale-balanced and the three summary
numbers (a ratio, a fraction, and an AUC) are all unitless and comparable.

Headline result (Norway PROBE stations = the deployed set the cold-start
paragraph is about; the held-out/test set gives near-identical numbers).
Numbers below are for the CURRENT latents selection (TESSERA v2 "1B-M" 2017,
crop64_lat16_auxon — see LATENTS_NPY below); v1-latents numbers in [brackets]:
  geographic   : isolation 44,   reachability  2%,  AUC 0.98  -> isolated
  elevation+mTPI: isolation 1.0,  reachability 95%,  AUC 0.61  -> reachable but
                  SUPERFICIAL (cannot tell Norway apart -> look-alike matches)
  ERA5 static  : isolation 3.3,  reachability 50%,  AUC 0.96  -> the baseline's
                  gridded static input: faithful but only partly reachable
  TESSERA lat16: isolation 1.2,  reachability 96%,  AUC 0.96  -> reachable AND
                  faithful: genuine surface analogues -> the model can transfer
                  a surface-matched correction.
                  [v1: isolation 1.3, reachability 79%, AUC 0.97 — the v2
                  embedding places nearly ALL of Norway within the training
                  support while staying just as faithful.]

Run:
    uv run python scripts/analysis/norway_descriptor_spaces.py
Outputs (the paper's Figs 11/12 (preprint) = F1/F2 (AMS) are re-rendered from
the same inputs by scripts/paper/make_paper_figures.py fig11/fig12):
    notebooks/norway_analysis_outputs/
        fig_descriptor_spaces.png            (Norway probe: isolation + AUC)
        fig_heldout_descriptor_control.png   (3 rollout groups: probe /
                                              held-out Norway / non-Norway
                                              control, reachability + AUC)
        fig_norway_reach_horizon.png         (horizon-resolved: reachability +
                                              AUC of the not-yet-deployed
                                              Norwegian stations vs the growing
                                              training set, over the rollout)
        fig_descriptor_clusters.png          (2-D projection per space: where
                                              Norway sits vs the training set ---
                                              overlapping / adjacent / isolated)
        descriptor_spaces_report.json        (all three groups, all metrics)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tessera_downscaling.paths import dataset_dir, processed_dir

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "norway_analysis_outputs"
OUT.mkdir(exist_ok=True, parents=True)
DATASET = dataset_dir("dataset_timestamp_global")
NB_LAT, NB_LON = (58.0, 71.0), (4.0, 31.0)

# ---- TESSERA latents generation (pick ONE) -----------------------------
# Current main: TESSERA v2 "1B-M", 2017 embeddings (crop64_lat16_auxon).
LATENTS_NPY = processed_dir(
    "vae_tessera_1B-M", "station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy"
)
# Previous main: TESSERA v1 16-d latents.
# LATENTS_NPY = processed_dir("station_latents_lat16_grad0.5.npy")

# ---- load & join latents + station table -------------------------------
lat = np.load(LATENTS_NPY)
ll = pd.read_csv(processed_dir("tessera_global", "station_list_filtered.csv"))
ll["station_id"] = ll["station_id"].astype(str)
lat_of = {s: i for i, s in enumerate(ll["station_id"])}
st = pd.read_csv(DATASET / "stations.csv")
st["station_id"] = st["station_id"].astype(str)
st["lrow"] = st["station_id"].map(lat_of)
st = st[st["lrow"].notna()].copy()
st["lrow"] = st["lrow"].astype(int)
st = st[~np.isnan(lat[st["lrow"].to_numpy()]).any(axis=1)].reset_index(drop=True)
Z16 = lat[st["lrow"].to_numpy()]

# ERA5 static interpolant: the coarse ERA5 static grid (the no-TESSERA
# baseline's gridded static input) bilinearly interpolated to each station.
# It is the per-station information the baseline conditions on *beyond*
# elevation+mTPI, so it belongs in the descriptor comparison. Stations are
# clipped to the grid bounds before interpolation (no extrapolation).
#
# We deliberately use the RAW interpolated static fields, NOT the model's
# CNN-encoded grid latent (interp_features). The encoded latent is dominated by
# the time-dependent DYNAMIC weather flowing through the same CNN, which is the
# field being *corrected* at the station rather than a persistent driver of the
# local correction; including it would conflate a station's transient weather
# state with its persistent surface character (Norway's distinctive climate
# alone makes it look OOD in the encoded latent). The descriptor question is
# whether a persistent surface analogue exists, so the time-invariant static
# input is the right space; the encoded-latent version is not used.
_eu_static = DATASET / "regions" / "europe"
_sfield = np.load(_eu_static / "static_fields.npy")  # (n_static, H, W)
_glat = np.load(_eu_static / "lats.npy")  # (H,) increasing
_glon = np.load(_eu_static / "lons.npy")  # (W,) increasing
_qpts = np.column_stack(
    [
        np.clip(st["latitude"].to_numpy(), _glat.min(), _glat.max()),
        np.clip(st["longitude"].to_numpy(), _glon.min(), _glon.max()),
    ]
)
ERA5_STATIC = np.column_stack(
    [
        RegularGridInterpolator(
            (_glat, _glon),
            _sfield[c],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )(_qpts)
        for c in range(_sfield.shape[0])
    ]
).astype(np.float32)

in_nb = (
    st.latitude.between(*NB_LAT)
    & st.longitude.between(*NB_LON)
    & (st.region == "europe")
).to_numpy()
eu = (st.region == "europe").to_numpy()
tr = (st.spatial_split == "train").to_numpy()
te = (st.spatial_split == "test").to_numpy()
groups = {
    "norway_test": in_nb & eu & te,
    "norway_probe": in_nb & eu & tr,
    "nonnorway_test": eu & te & ~in_nb,  # the held-out "control" group
    "rest_train": eu & tr & ~in_nb,
}

# Hand-crafted land-surface descriptor: the 17-feature GEE vector built by
# scripts/preprocessing/build_extra_descriptors.py — WorldCover class
# fractions, tree height, soil clay/sand, elevation mean/std/min/max, slope and
# directional gradients. Row-aligned with the same station_list_filtered.csv
# the latents use, so `lrow` indexes it directly.
#
# This is the space that answers the sharpest objection to the paper: that the
# embedding only wins because the hand-crafted baseline is impoverished. Giving
# the comparison an explicit land-cover / roughness / terrain-heterogeneity
# descriptor puts that hypothesis on the same footing as every other space
# here, model-free.
#
# OPT-IN. The paper's §3.5 figures compare the four spaces below and nothing
# else, so these two extra spaces are gated behind an environment variable:
# a default run reproduces the published figures unchanged. Enable with
#     WITH_EXTRA_DESCRIPTORS=1 uv run python scripts/analysis/norway_descriptor_spaces.py
# for the appendix version.
_EXTRA_NPY = processed_dir("extra_descriptors.npy")
_WITH_EXTRA = os.environ.get("WITH_EXTRA_DESCRIPTORS", "0") not in ("0", "", "false")
EXTRA_DESC = None
if not _WITH_EXTRA:
    print(
        "[extra descriptors] disabled (set WITH_EXTRA_DESCRIPTORS=1 to include "
        "the hand-crafted land-cover + terrain spaces)"
    )
elif _EXTRA_NPY.exists():
    _extra_all = np.load(_EXTRA_NPY)
    _extra = _extra_all[st["lrow"].to_numpy()]
    _n_nan_rows = int(np.isnan(_extra).any(axis=1).sum())
    if _n_nan_rows:
        # Mean-fill rather than dropping stations: dropping would change the
        # station set for this space alone, making the bars non-comparable.
        _col_mean = np.nanmean(_extra, axis=0)
        _extra = np.where(np.isnan(_extra), _col_mean, _extra)
        print(f"[extra descriptors] mean-filled {_n_nan_rows} NaN rows")
    EXTRA_DESC = _extra.astype(np.float32)
    print(f"[extra descriptors] loaded {EXTRA_DESC.shape} from {_EXTRA_NPY.name}")
else:
    print(f"[extra descriptors] {_EXTRA_NPY} not found — skipping those spaces")

SPACES = {
    "geographic\n(lat, lon)": st[["latitude", "longitude"]].to_numpy(),
    "elevation\n+ mTPI": st[["elevation", "delta_elevation", "mtpi"]].to_numpy(),
    "ERA5 static\n(interp)": ERA5_STATIC,
}
COLOUR = {
    "geographic\n(lat, lon)": "#7f7f7f",
    "elevation\n+ mTPI": "#ff7f0e",
    "ERA5 static\n(interp)": "#9467bd",
    "land surface\n(hand-crafted)": "#8c564b",
    "elev+mTPI\n+ land surface": "#c49a6c",
    "TESSERA\nlat16 embedding": "#1f77b4",
}

if EXTRA_DESC is not None:
    # Two variants: the hand-crafted land-surface features on their own (the
    # direct counterpart to TESSERA, both being surface descriptors), and
    # stacked with elevation+mTPI — which is exactly the descriptor the
    # `*_extradesc_*` ConvCNP arm receives, so the model row and this
    # model-free row measure the same input.
    SPACES["land surface\n(hand-crafted)"] = EXTRA_DESC
    SPACES["elev+mTPI\n+ land surface"] = np.column_stack(
        [
            st[["elevation", "delta_elevation", "mtpi"]].to_numpy(),
            EXTRA_DESC,
        ]
    )

# TESSERA last so it reads as the rightmost bar in every panel.
SPACES["TESSERA\nlat16 embedding"] = Z16


def metrics(X: np.ndarray, query_mask: np.ndarray) -> dict:
    """Isolation factor, reachability, separability AUC for `query` vs rest_train."""
    scaler = StandardScaler().fit(X[groups["rest_train"]])
    Xs = scaler.transform(X)
    ref = Xs[groups["rest_train"]]
    q = Xs[query_mask]
    # in-distribution scale: rest->rest leave-one-out NN
    d_loo, _ = NearestNeighbors(n_neighbors=2).fit(ref).kneighbors(ref)
    d_loo = d_loo[:, 1]
    base = float(np.median(d_loo))
    r95 = float(np.percentile(d_loo, 95))
    # query -> nearest rest-train
    dq, _ = NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(q)
    dq = dq[:, 0]
    # separability AUC (Norway vs rest-train)
    Xa = np.vstack([q, ref])
    ya = np.r_[np.ones(len(q)), np.zeros(len(ref))]
    auc = float(
        cross_val_score(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
            Xa,
            ya,
            cv=StratifiedKFold(4, shuffle=True, random_state=0),
            scoring="roc_auc",
        ).mean()
    )
    return {
        "isolation": float(np.median(dq) / base),
        "reachability": float(np.mean(dq <= r95)),
        "auc": auc,
        "median_nn": float(np.median(dq)),
        "indist_nn": base,
        "n_query": int(query_mask.sum()),
        "n_ref": int(groups["rest_train"].sum()),
    }


report = {
    q: {s: metrics(X, groups[q]) for s, X in SPACES.items()}
    for q in ("norway_test", "norway_probe", "nonnorway_test")
}
(OUT / "descriptor_spaces_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["norway_test"], indent=2))

# ---- figure (headline = deployed Norway PROBE stations) ----------------
# The cold-start paragraph is about the probe set; the held-out/test set
# (report["norway_test"]) gives near-identical numbers.
R = report["norway_probe"]
names = list(SPACES)
cols = [COLOUR[n] for n in names]
iso = [R[n]["isolation"] for n in names]
reach = [R[n]["reachability"] * 100 for n in names]
auc = [R[n]["auc"] for n in names]

fig, ax = plt.subplots(1, 2, figsize=(2.9 * len(SPACES), 4.8))

# Panel A: isolation factor (log), annotated with reachability %
bars = ax[0].bar(range(len(names)), iso, color=cols, edgecolor="black", linewidth=0.6)
ax[0].set_yscale("log")
ax[0].axhline(1.0, color="black", ls=":", lw=1, label="in-distribution (=1)")
ax[0].set_ylabel(
    "isolation factor\n(NN$_{\\mathrm{Norway\\to rest}}$ / NN$_{\\mathrm{rest\\to rest}}$, log)"
)
ax[0].set_title(
    "(a) How far is Norway from European training\nsurfaces? (lower = more reachable)"
)
for i, (b, r) in enumerate(zip(bars, reach, strict=False)):
    ax[0].text(
        i,
        b.get_height() * 1.15,
        f"×{iso[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax[0].text(
        i,
        min(iso) * 0.55,
        f"{r:.0f}% reachable",
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )
ax[0].set_xticks(range(len(names)))
ax[0].set_xticklabels(names, fontsize=9)
ax[0].set_ylim(min(iso) * 0.35, max(iso) * 2.2)
ax[0].legend(fontsize=8, loc="upper right")

# Panel B: faithfulness (separability AUC)
bars = ax[1].bar(range(len(names)), auc, color=cols, edgecolor="black", linewidth=0.6)
ax[1].axhline(0.5, color="black", ls=":", lw=1, label="chance (0.5)")
ax[1].set_ylim(0.5, 1.02)
ax[1].set_ylabel("separability AUC\n(Norway vs rest-EU-train)")
ax[1].set_title(
    "(b) Does the descriptor faithfully\ndistinguish Norway? (higher = more faithful)"
)
for i, b in enumerate(bars):
    ax[1].text(
        i,
        b.get_height() + 0.008,
        f"{auc[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, fontsize=9)
ax[1].legend(fontsize=8, loc="lower left")

fig.suptitle(
    "Norway is reachable from European training surfaces only via a faithful descriptor.\n"
    "Geography isolates it (a); elevation+mTPI make it look reachable (a) but cannot tell it apart "
    "(b, superficial);\nthe baseline's ERA5 static input is faithful (b) but only partly reachable (a); "
    "TESSERA is both reachable (a) and faithful (b).",
    fontsize=10.5,
    y=1.02,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig_descriptor_spaces.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nWrote", OUT / "fig_descriptor_spaces.png")

# ---- descriptor figure over the THREE rollout station groups -----------
# Same partition as the rollout-gap figure (data_efficiency_temporal_rollout):
# deployed Norway probes, held-out Norway, and the held-out non-Norway control.
# Two messages in one figure:
#   1. The two Norway groups (probe / held-out) are near-identical in EVERY
#      space -> reachability/faithfulness is a property of a station's
#      LOCATION, not of whether it was trained on. So the descriptor analysis
#      applies to the probe paragraph and the held-out paragraph alike.
#   2. The non-Norway control is in-distribution by EVERY descriptor
#      (AUC~0.5, ~95% reachable) -> both models interpolate it and it stays
#      flat over the rollout. Norway is the lone exception: geo-isolated and a
#      distinct embedding corner, only superficially matched by elevation.
CT = {
    "Norway — newly deployed (probe)": report["norway_probe"],
    "Norway — held-out": report["norway_test"],
    "non-Norway — held-out (control)": report["nonnorway_test"],
}
QCOL = {
    "Norway — newly deployed (probe)": "#1f77b4",
    "Norway — held-out": "#d62728",
    "non-Norway — held-out (control)": "#2ca02c",
}
space_names = list(SPACES)
xx = np.arange(len(space_names))
w = 0.27
offs = {0: -w, 1: 0.0, 2: +w}

fig, ax = plt.subplots(1, 2, figsize=(3.1 * len(SPACES), 5.2))
# Panel A: reachability (% with an in-distribution analogue)
for k, (qn, r) in enumerate(CT.items()):
    vals = [r[s]["reachability"] * 100 for s in space_names]
    bars = ax[0].bar(
        xx + offs[k],
        vals,
        w,
        color=QCOL[qn],
        edgecolor="black",
        linewidth=0.6,
        label=qn,
    )
    for b, v in zip(bars, vals, strict=False):
        ax[0].text(
            b.get_x() + b.get_width() / 2,
            v + 1.5,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
ax[0].set_ylim(0, 108)
ax[0].set_ylabel("% of stations with an\nin-distribution analogue")
ax[0].set_title("(a) Reachable? (higher = more in-distribution)")
ax[0].set_xticks(xx)
ax[0].set_xticklabels(space_names, fontsize=9)

# Panel B: separability AUC (in-distribution iff ~0.5)
for k, (qn, r) in enumerate(CT.items()):
    vals = [r[s]["auc"] for s in space_names]
    bars = ax[1].bar(
        xx + offs[k],
        vals,
        w,
        color=QCOL[qn],
        edgecolor="black",
        linewidth=0.6,
        label=qn,
    )
    for b, v in zip(bars, vals, strict=False):
        # Keep every value label clear of the 0.5 reference line; labels for
        # sub-0.5 bars would otherwise be bisected by it.
        ly = v + 0.012 if v >= 0.52 else 0.515
        ax[1].text(
            b.get_x() + b.get_width() / 2,
            ly,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
# The dotted line is identified by the panel title ("0.5 = in-distribution")
# and the y-axis tick, so it needs no in-axes annotation.
ax[1].axhline(0.5, color="black", ls=":", lw=1)
ax[1].set_ylim(0.4, 1.08)
ax[1].set_ylabel("separability AUC\n(group vs training)")
ax[1].set_title("(b) Distinguishable from training?\n(0.5 = in-distribution)")
ax[1].set_xticks(xx)
ax[1].set_xticklabels(space_names, fontsize=9)

# No suptitle: the interpretation lives in the LaTeX caption. Keep the
# group legend (the only on-figure key the reader needs).
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    fontsize=8.5,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.99),
    frameon=True,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig_heldout_descriptor_control.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote", OUT / "fig_heldout_descriptor_control.png")

# ---- deployment-resolved reachability: Norway becoming in-distribution ----
# The static figures above are a snapshot; this one resolves the rollout,
# indexed by the NUMBER of Norwegian probe stations deployed into training. At
# each snapshot we take the Norwegian stations NOT IN THE CURRENT TRAINING SET
# -- the not-yet-deployed probes plus the permanently held-out test stations,
# the residual spatial-extrapolation target -- and ask, against that training
# set (non-Norway Europe plus the Norwegian stations deployed so far), what
# fraction has an in-distribution analogue (a) and how separable they remain
# (b). As deployment completes the probes drop out of this query and only the
# held-out test set remains. The reference scale (z-scoring + R95) is fixed on
# the non-Norway European train, the same stable frame as above. This makes
# explicit why the probe and held-out curves diverge: as Norway fills in, the
# geographic descriptor carries Norway from extrapolation toward interpolation,
# whereas the embedding already places ~80% of Norway in-distribution from the
# very first stations -- reachable AND faithful when geography is neither.
ROLLOUT_FOLDER = "snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi"
SCHED = REPO / "scripts" / "experiments" / ROLLOUT_FOLDER / "rollout_schedule.json"
# Through full deployment: at r3y all 1505 probes are online, so the query is
# exactly the held-out test set; r4y..r6y add no stations and are omitted.
HORIZON_ORDER = ["r1mo", "r3mo", "r6mo", "r1y", "r2y", "r3y"]
LABEL = {
    "geographic\n(lat, lon)": "geographic",
    "elevation\n+ mTPI": "elevation + mTPI",
    "ERA5 static\n(interp)": "ERA5 static (interp)",
    "TESSERA\nlat16 embedding": "TESSERA embedding",
}

if not SCHED.exists():
    print(f"(no rollout_schedule.json at {SCHED} -> skipping horizon figure)")
else:
    sched = json.loads(SCHED.read_text())
    st_id = st["station_id"].to_numpy()
    norway_probe = groups["norway_probe"]

    # Fixed reference frame per space: scaler on rest_train, R95 from its LOO NN.
    prep = {}
    for name, X in SPACES.items():
        sc = StandardScaler().fit(X[groups["rest_train"]])
        Xs = sc.transform(X)
        d_loo = (
            NearestNeighbors(n_neighbors=2)
            .fit(Xs[groups["rest_train"]])
            .kneighbors(Xs[groups["rest_train"]])[0][:, 1]
        )
        prep[name] = (Xs, float(np.percentile(d_loo, 95)))

    def _reach_auc_rows(deployed):
        """Reachability % and separability AUC per space for a deployed mask.

        Query = Norwegian stations NOT in the current training set (the
        not-yet-deployed probes plus the permanently held-out test set); at
        zero deployment this is all of Norway, at full deployment only the
        held-out test set. Reference = non-Norway train plus deployed probes.
        """
        out_of_training = (norway_probe & ~deployed) | groups["norway_test"]
        ref_mask = groups["rest_train"] | deployed
        n_dep = int(deployed.sum())  # x-axis: # Norwegian probes in training
        rows = []
        for name, (Xs, r95) in prep.items():
            q = Xs[out_of_training]
            ref = Xs[ref_mask]
            dq = NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(q)[0][:, 0]
            reach = float(np.mean(dq <= r95)) * 100
            Xa = np.vstack([q, ref])
            ya = np.r_[np.ones(len(q)), np.zeros(len(ref))]
            auc = float(
                cross_val_score(
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"),
                    ),
                    Xa,
                    ya,
                    cv=StratifiedKFold(4, shuffle=True, random_state=0),
                    scoring="roc_auc",
                ).mean()
            )
            rows.append(dict(x=n_dep, space=name, reach=reach, auc=auc))
        return rows

    hz = []
    # Cold start (x=0): zero Norwegian stations deployed, so the query is all
    # of Norway against the non-Norway European train alone -- the true
    # pre-deployment snapshot, the same state the cluster figure reports.
    hz.extend(_reach_auc_rows(np.zeros(len(st), dtype=bool)))
    for hlabel in HORIZON_ORDER:
        sp = sched["sweep_points"][hlabel]
        deployed_ids = {
            sid
            for sid, v in sp["probe_active_from"].items()
            if not str(v).startswith("9999")
        }
        deployed = norway_probe & np.isin(st_id, list(deployed_ids))
        hz.extend(_reach_auc_rows(deployed))
    hz = pd.DataFrame(hz)

    fig, ax = plt.subplots(1, 2, figsize=(3.1 * len(SPACES), 4.8))
    for name in SPACES:
        s = hz[hz["space"] == name].sort_values("x")
        ax[0].plot(
            s["x"],
            s["reach"],
            marker="o",
            color=COLOUR[name],
            lw=1.8,
            label=LABEL[name],
        )
        ax[1].plot(
            s["x"], s["auc"], marker="o", color=COLOUR[name], lw=1.8, label=LABEL[name]
        )
    # No panel titles and a single legend (panel a): the panel reading and the
    # 0.5-line meaning live in the LaTeX caption; the curves are the same
    # spaces in both panels.
    ax[0].set_ylim(-5, 108)
    ax[0].set_ylabel(
        "% of out-of-training Norwegian stations\n"
        "with an in-distribution training analogue"
    )
    ax[0].set_xlabel("Norwegian probe stations deployed into training")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8.5, loc="lower right")
    ax[1].axhline(0.5, color="black", ls=":", lw=1)
    ax[1].set_ylim(0.45, 1.0)
    ax[1].set_ylabel("separability AUC\n(out-of-training Norway vs current training)")
    ax[1].set_xlabel("Norwegian probe stations deployed into training")
    ax[1].grid(alpha=0.3)
    # Mark the cold-start (x=0) column: no Norwegian station deployed yet.
    # Light dotted guide with the label parked in the bottom margin, at the
    # foot of the line and below every curve so it never crosses one.
    for a in ax:
        a.axvline(0, color="#bbbbbb", ls=":", lw=1.0, zorder=0)
    ax[0].text(
        15, -3, "cold start", fontsize=8, color="#666666", va="center", ha="left"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_norway_reach_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT / "fig_norway_reach_horizon.png")

# ---- cluster projection: where each descriptor places Norway --------------
# A 2-D view of each space that makes the coverage-vs-shift distinction visible.
# x = the Norway-vs-rest linear discriminant (the axis the separability AUC
# looks along); y = the leading residual principal component. Elevation+mTPI
# collapses Norway onto the rest of Europe (indistinguishable even on its own
# best-separating axis -> its "reachability" is spurious look-alikes), whereas
# geography, the ERA5 static fields, and the embedding place Norway as a
# distinct but ADJACENT cluster (distributionally shifted, yet most members
# still sit within the training set's support -> genuine analogues). The
# non-Norway held-out control overlaps the training set in every space --- the
# genuinely in-distribution case.
_norway = groups["norway_probe"] | groups["norway_test"]
_rng = np.random.RandomState(0)
_sub = _rng.choice(
    np.where(groups["rest_train"])[0],
    size=min(2500, int(groups["rest_train"].sum())),
    replace=False,
)
_sub_mask = np.zeros(len(st), bool)
_sub_mask[_sub] = True

fig, ax = plt.subplots(1, len(SPACES), figsize=(5 * len(SPACES), 5))
for a, (name, X) in zip(ax, SPACES.items(), strict=False):
    Xs = StandardScaler().fit(X[groups["rest_train"]]).transform(X)
    if name.startswith("geographic"):
        P = Xs[:, [1, 0]]
        xl, yl = "longitude (z)", "latitude (z)"
    else:
        ya = np.r_[np.ones(_norway.sum()), np.zeros(groups["rest_train"].sum())]
        Xa = np.vstack([Xs[_norway], Xs[groups["rest_train"]]])
        w = LinearDiscriminantAnalysis().fit(Xa, ya).scalings_[:, 0]
        w = w / np.linalg.norm(w)
        px = Xs @ w
        resid = Xs - np.outer(px, w)
        py = PCA(n_components=1).fit(resid[groups["rest_train"]]).transform(resid)[:, 0]
        P = np.column_stack([px, py])
        xl, yl = "Norway-vs-rest discriminant", "residual PC1"
    a.scatter(
        P[_sub_mask, 0],
        P[_sub_mask, 1],
        s=4,
        c="#cccccc",
        label="rest-train",
        rasterized=True,
    )
    a.scatter(
        P[groups["nonnorway_test"], 0],
        P[groups["nonnorway_test"], 1],
        s=6,
        c="#2ca02c",
        alpha=0.6,
        label="non-Norway held-out (control)",
    )
    a.scatter(P[_norway, 0], P[_norway, 1], s=6, c="#d62728", alpha=0.6, label="Norway")
    _r = report["norway_probe"][name]
    a.set_title(
        f"{name.replace(chr(10), ' ')}\n"
        f"reachable {_r['reachability'] * 100:.0f}%,  AUC {_r['auc']:.2f}",
        fontsize=10,
    )
    a.set_xlabel(xl, fontsize=9)
    a.set_ylabel(yl, fontsize=9)
    a.tick_params(labelsize=8)
    if a is ax[0]:
        a.legend(fontsize=8, markerscale=2, loc="best")
# No suptitle: the interpretation lives in the LaTeX caption.
fig.tight_layout()
fig.savefig(OUT / "fig_descriptor_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote", OUT / "fig_descriptor_clusters.png")
