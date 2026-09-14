# Aurora latent extraction on CSD3 (MI355X)

How the workstation drives `scripts/aurora/extract_aurora_latents.py` on
CSD3, one chunk of valid times at a time, and pulls the result back. Code
only lives here; the workstation session issues every execution step over
ssh and owns the data on `/data/weather-downscaling`.

## Layout on CSD3

| what | where |
|---|---|
| repo | `~/tessera-downscaling` on the login nodes (branch `feat/aurora-latents`) |
| RDS root (`RDS_ROOT`) | `/rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling` (project quota: 15 TB, ~3.4 TB free in Sept 2026, shared with other users) |
| ROCm env | `${RDS_ROOT}/env/aurora-rocm` (`setup_csd3_env.sh`; python 3.12, torch ROCm wheels, `microsoft-aurora`, this repo editable) |
| inputs (scp'd once) | `${RDS_ROOT}/datasets/dataset_timestamp_global/{metadata.json,regions/{europe,east_asia}/{lats,lons}.npy}`, `${RDS_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc` |
| ERA5 staging (transient) | `${RDS_ROOT}/ingest/aurora_inputs/era5_wb2_quarter_<var>/data/<ts>.nc` |
| latents | `${RDS_ROOT}/ingest/aurora/lead{6,24,72}h/<region>/latent_backbone/`, `lead0h/<region>/latent_encoder/`, `latent_calibration.json` |
| verification subset | `${RDS_ROOT}/ingest/aurora_verify/lead{L}h/<region>/processed/...` |
| markers | `${RDS_ROOT}/ingest/aurora/chunks/<id>.{staged,done,sha256,verify.txt,pulled,cleaned}` |
| job logs | `~/tessera-downscaling/logs/aurora/<chunk>/` |

Slurm: GPU jobs run on `mi355x` (account `zea-p005-zenith-gpu`, QOS `gpu1`,
36 h wall, one GPU per job, 8 GPUs per node so shards share nodes); CPU jobs
(staging download, verification) on `desktop9` (account `zea-p005-cpu`, QOS
`cpu1`, 4 cores / 64 GB per node). Compute nodes reach GCS and HuggingFace
directly (~250 MB/s measured with 16 streams). All of this is set in
`scripts/aurora/_csd3.sh`.

The GPU nodes have no system ROCm, only the amdgpu driver; the wheels bring
the runtime. **The project's torch pin (2.9.0) does not run on the MI355X**:
its ROCm 6.4 runtime rejects gfx950, so the env uses torch 2.10.0 from the
`rocm7.0` index (verified by `setup_csd3_env.sh --gpu-check`, which runs the
small Aurora model on the GPU). The extractor records the torch version in
every `latent_meta.json`.

## Chunks

`scripts/aurora/chunks.json` (helper: `chunks.py list|show|stems|check`).
One chunk = one calendar year of valid times (`2010` .. `2022`; the last one
runs to 2023-01-10-18), 13 chunks, 19,032 slots in total. Staging window =
`[valid_start - 78 h, valid_end]`. Per chunk: ~430 GB of 13-level inputs,
~230 GB of fp16 latents (backbone 29 MB + encoder 15 MB per europe file, 16 +
8 MB per east_asia file), ~1,470 rollouts of 12 steps. One-year chunks keep
"inputs of chunk N+1 + latents of chunk N" inside the free quota, so the next
chunk can be staged while the current one is extracted or pulled.

Consecutive chunks share the 12 inits before a chunk's first valid time
(their short leads belong to the previous chunk): `clean` keeps the shared
encoder files and staging frames until the next chunk's `clean`.

## One-off preparation

```bash
# workstation: the 12 MB inputs tarball, laid out as above, to the RDS root
scp inputs.tar.gz csd3:/rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling/
ssh csd3 'cd /rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling && tar xzf inputs.tar.gz'

ssh csd3 'cd ~/tessera-downscaling && git fetch && git checkout feat/aurora-latents && git pull'
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/setup_csd3_env.sh --tests --gpu-check'   # idempotent
```

## Per-chunk commands (issued from the workstation)

The submitter: `bash scripts/aurora/submit_aurora_latents.sh --chunk <id> --mode <mode>`
(all modes: `--help`; `--print` shows the sbatch commands without submitting).

```bash
# 0. what is there
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode status'

# 1. stage the chunk's inputs (CPU array + check job -> chunks/2010.staged)
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode stage'

# 2. once staged: the extractor's own pre-flight on the login node (no job)
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode dry-run'

# 3. smoke test on a GPU (small model, 3 inits, --verify-decoder) into ingest/aurora_smoke
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode smoke'

# 4. calibration, once, on chunk 2010 -> ingest/aurora/latent_calibration.json
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode calibrate'

# 5. the extraction: array of 4 contiguous sub-windows, one MI355X each
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode shard --n-shards 4'

# 6. structural verification -> chunks/2010.done + chunks/2010.sha256
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode verify'
```

Steps 1, 4, 5, 6 chained with Slurm dependencies in one command:

```bash
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2010 --mode pipeline --with-calibrate'
ssh csd3 'cd ~/tessera-downscaling && bash scripts/aurora/submit_aurora_latents.sh --chunk 2011 --mode pipeline'
```

`--after <jobid>` chains any mode behind another job (e.g. stage 2011 after
the verify of 2010; a shard array after a calibrate job). A dependent job
whose predecessor fails is cancelled (`--kill-on-invalid-dep`), so a failed
chunk shows up as cancelled jobs in `status`, never as a half-run.

Then, on the workstation (waits for the marker, rsyncs without `--delete`,
checks the manifest, runs the verifier with the drift check, marks the chunk
pulled and cleans it on RDS):

```bash
bash scripts/aurora/pull_aurora_chunk.sh --chunk 2010            # --no-clean to keep it on RDS
```

Equivalent manual steps: poll `test -f ${RDS_ROOT}/ingest/aurora/chunks/2010.done`;
`rsync -a` the `lead*h/*/latent_*` directories, `latent_calibration.json`,
`chunks/` and `ingest/aurora_verify/` into `/data/weather-downscaling/ingest/`;
`(cd /data/weather-downscaling && sha256sum -c ingest/aurora/chunks/2010.sha256)`;
`verify_aurora_latents.py --start-date 2010-01-01-00 --end-date 2010-12-31-18`;
`touch .../chunks/2010.pulled` on RDS; `--mode clean`.

## Resume and failure handling

* Every step skips what exists: the downloader (`atomic_completed`), the
  extractor (per-file resume; an init whose files are all present is not
  rolled out again), `verify` and `stage` rewrite their markers.
  Re-issuing the same command after a failure continues where it stopped.
* `stage`'s check job fails (and writes no `.staged`) if any download left a
  `-problem.txt` file, after removing the partial output; resubmit `stage`.
* A failed shard array task: `--mode shard` again (the surviving files are
  kept; only the missing ones are recomputed). `--mode verify` reports
  exactly which timestamps are missing.
* `shard` refuses to run without the calibration file, and `clean` refuses
  without both `.done` and `.pulled` (`--force` overrides, deliberately).
* The smoke test writes to `ingest/aurora_smoke` (+ `_verify`), never to the
  real tree: small-model files there would be taken for finished latents.
* The structural verifier on CSD3 is run with a non-existent `--verify-root`
  on purpose: with the decoded subset present but no originals it would fail
  the drift comparison; the drift check runs on the workstation.

## Sizing (per one-year chunk, MI355X)

Fill in after chunk 2010: seconds per rollout step from the calibration
output (`latent_calibration.json: seconds_per_step`), shard wall time from
`logs/aurora/2010/shard_*.out`, staging wall time from `stage_*.out`. Shard
jobs ask for 24 h (`SHARD_TIME`), staging 12 h (`STAGE_TIME`); raise
`--n-shards` if a year's shard runs longer than that.
