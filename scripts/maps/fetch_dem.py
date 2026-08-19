"""Build a per-cell high-res elevation array for a region's 0.05 deg dense grid.

Downloads SRTM 1-arcsec (~30 m) .hgt tiles from the public AWS terrain-tiles
bucket (no auth, no rasterio needed), point-samples each grid-cell centroid
(bilinear) — matching how the model was trained on station POINT elevations —
and saves an array aligned with the latent grid (grid_idx order).

Region selected by the REGION env var (default "iberia"). Output:
<data_root>/processed/dense/<region>/<region>_0.05deg_dem.npy  (N,) metres.
Tiles are cached under <data_root>/processed/dem_cache/ so re-runs are instant.

  REGION=norway uv run python scripts/maps/fetch_dem.py
"""

import gzip
import math
import urllib.request

import numpy as np
from regions import get_region

from tessera_downscaling.paths import processed_dir

R = get_region()
NPZ = R.dense_npz
OUT = R.dem_path
CACHE = processed_dir("dem_cache")
CACHE.mkdir(parents=True, exist_ok=True)
BASEURL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"


def tile_name(tlat, tlon):
    ns = f"N{tlat:02d}" if tlat >= 0 else f"S{-tlat:02d}"
    ew = f"E{tlon:03d}" if tlon >= 0 else f"W{-tlon:03d}"
    return ns, ew


def load_tile(tlat, tlon):
    ns, ew = tile_name(tlat, tlon)
    fn = CACHE / f"{ns}{ew}.hgt"
    if not fn.exists():
        url = f"{BASEURL}/{ns}/{ns}{ew}.hgt.gz"
        try:
            raw = gzip.decompress(urllib.request.urlopen(url, timeout=90).read())
        except Exception as e:
            print("  MISS", ns + ew, type(e).__name__)
            return None
        fn.write_bytes(raw)
    raw = fn.read_bytes()
    n = int((len(raw) / 2) ** 0.5)
    a = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)
    a[a < -1000] = np.nan  # SRTM voids / sea
    return a, n


def main():
    d = np.load(NPZ, allow_pickle=True)
    co = d["coords"]
    vm = d["valid_mask"]
    lat = co["lat"].astype(float)
    lon = co["lon"].astype(float)
    N = len(lat)
    dem = np.full(N, np.nan, np.float32)

    tiles = {}
    for i in range(N):
        key = (int(math.floor(lat[i])), int(math.floor(lon[i])))
        tiles.setdefault(key, []).append(i)

    for (tlat, tlon), idxs in sorted(tiles.items()):
        idxs = np.array(idxs)
        if not vm[idxs].any():  # skip ocean-only tiles
            continue
        res = load_tile(tlat, tlon)
        if res is None:
            continue
        a, n = res
        plat, plon = lat[idxs], lon[idxs]
        fy = np.clip((tlat + 1 - plat) * (n - 1), 0, n - 1 - 1e-6)
        fx = np.clip((plon - tlon) * (n - 1), 0, n - 1 - 1e-6)
        y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
        wy, wx = fy - y0, fx - x0
        c = np.stack([a[y0, x0], a[y0, x0 + 1], a[y0 + 1, x0], a[y0 + 1, x0 + 1]], 0)
        val = (
            c[0] * (1 - wy) * (1 - wx)
            + c[1] * (1 - wy) * wx
            + c[2] * wy * (1 - wx)
            + c[3] * wy * wx
        )
        bad = ~np.isfinite(val)
        if bad.any():
            val[bad] = np.nanmean(c[:, bad], axis=0)  # fall back to valid corners
        dem[idxs] = val
        print(
            f"  tile {tlat:+03d},{tlon:+04d} cells={len(idxs):4d} elev[{np.nanmin(val):.0f},{np.nanmax(val):.0f}]"
        )

    np.save(OUT, dem)
    print(
        f"SAVED {OUT.name}: {int(np.isfinite(dem).sum())}/{N} cells, "
        f"valid-land dem max {np.nanmax(np.where(vm, dem, np.nan)):.0f} m"
    )


if __name__ == "__main__":
    main()
