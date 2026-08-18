"""Migrate legacy ConvCNP checkpoints to the v4 per-variable heads layout.

Legacy checkpoints (saved before the per-variable likelihood heads
abstraction landed) had a single shared output projection in the decoder
MLP:

    DecoderMLP:    `mlp.net.<N>.weight`     of shape ``[2V, H]``
                   `mlp.net.<N>.bias`        of shape ``[2V]``
    FiLMDecoderMLP: `mlp.output_layer.weight` of shape ``[2V, H]``
                    `mlp.output_layer.bias`   of shape ``[2V]``

where ``V == n_target_variables`` and the output was packed row-wise as
``(mean_0, log_var_0, mean_1, log_var_1, …, mean_{V-1}, log_var_{V-1})``.

The v4 model removes the shared projection from the body and gives each
variable its own ``Linear(H, n_params)`` inside a ``LikelihoodHeadDict``.
For Gaussian heads (the only kind any legacy checkpoint uses), each
per-variable Linear has shape ``[2, H]`` with the same row order
``(mean, log_var)``. Migration is therefore a row-wise tensor split, key
rename, and an ``[N, H]`` body tensor pass-through:

    legacy:   `mlp.net.{N}.{weight,bias}`   shape [2V, ...] / [2V]
    →
    new:      `heads.heads.<var_i>.linear.{weight,bias}`   shape [2, ...] / [2]
              for i in 0 … V-1, with rows [2i, 2i+1] of the legacy tensor.

The migration is provably correct by construction (it preserves every
byte of the underlying ``nn.Linear`` arithmetic), and the script
verifies this on every checkpoint via a tensor-equality round-trip
check before renaming files on disk.

This script is idempotent: re-running it on a checkpoint that has
already been migrated is a no-op (checkpoints carrying the
``heads.heads.*`` keys already are skipped with a warning).

Hypernet checkpoints are refused — the hypernet variant generated the
entire translator including its output layer, leaving no static weight
tensor to split. Pre-v4 hypernet experiments are no longer accessible
under the new code path; the original .pt files are preserved untouched.

USAGE
-----

  # Migrate every checkpoint under a folder of training runs.
  python scripts/migrate_legacy_checkpoints.py path/to/training_runs/

  # Dry-run: report what would change without touching anything.
  python scripts/migrate_legacy_checkpoints.py path/to/training_runs/ --dry-run

After migration, each run directory will have:

  - best_model.pt              — migrated checkpoint, loadable by v4 code.
  - best_model.legacy.pt       — original, untouched (full backup).
  - latest_model.pt            — also migrated, with .legacy.pt backup.
  - config.json                — augmented with ``likelihood_per_variable``
                                 (always ``{var: "gaussian"}`` because
                                 every legacy checkpoint was Gaussian-only).
  - config.legacy.json         — original config.

The ``.legacy.*`` backups are explicitly left in place and are NOT used
by v4 code; rolling back a migration is therefore as simple as moving
the .legacy file back into place.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import torch


logger = logging.getLogger("migrate_legacy_checkpoints")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _is_already_migrated(state_dict: dict) -> bool:
    """A v4 checkpoint will have at least one ``heads.heads.*`` key."""
    return any(k.startswith("heads.heads.") for k in state_dict)


def _detect_legacy_final_layer_keys(
    state_dict: dict, n_target_variables: int, tessera_injection: str,
) -> tuple[str, str]:
    """Locate the legacy final projection's weight and bias keys.

    Two layouts:

      ``DecoderMLP`` (concat / no TESSERA): the body is ``self.net``,
      a ``nn.Sequential`` of (Linear, ReLU, Linear, ReLU, …, Linear).
      The final Linear is the one whose weight tensor has first
      dimension ``2 * V``. The script searches over even-indexed
      positions (``mlp.net.0``, ``mlp.net.2``, ``mlp.net.4``, …).

      ``FiLMDecoderMLP`` (FiLM): the final Linear is named
      ``mlp.output_layer`` directly.

    Returns ``(weight_key, bias_key)``.
    """
    if tessera_injection == "film":
        weight_key = "mlp.output_layer.weight"
        bias_key = "mlp.output_layer.bias"
        if weight_key not in state_dict:
            raise KeyError(
                f"Expected legacy FiLM checkpoint to contain "
                f"{weight_key!r}, but it's not in the state dict. "
                f"Either this is not a FiLM checkpoint or the layout "
                "has changed unexpectedly."
            )
        return weight_key, bias_key

    # Concat (or no-TESSERA) path: search through mlp.net.<N>.weight keys.
    expected_first_dim = 2 * n_target_variables
    candidates = []
    pattern = re.compile(r"^mlp\.net\.(\d+)\.weight$")
    for k, v in state_dict.items():
        m = pattern.match(k)
        if m and v.dim() >= 1 and v.shape[0] == expected_first_dim:
            candidates.append((int(m.group(1)), k))

    if not candidates:
        raise KeyError(
            f"Could not find a legacy final projection in `mlp.net.*.weight` "
            f"with first dim {expected_first_dim}. Looked at: "
            f"{[k for k in state_dict if k.startswith('mlp.net.')]}"
        )
    # Final Linear is the highest-indexed one (later in the Sequential).
    candidates.sort(key=lambda x: x[0])
    weight_key = candidates[-1][1]
    bias_key = weight_key.replace(".weight", ".bias")
    return weight_key, bias_key


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate_state_dict(
    legacy_state: dict, target_variables: list[str], tessera_injection: str,
) -> tuple[dict, str, str]:
    """Apply the row-wise split + key rename.

    Returns ``(new_state_dict, legacy_weight_key, legacy_bias_key)``. The
    two extra return values are used by the round-trip check.
    """
    weight_key, bias_key = _detect_legacy_final_layer_keys(
        legacy_state, len(target_variables), tessera_injection,
    )
    legacy_W = legacy_state[weight_key]
    legacy_b = legacy_state[bias_key]
    V = len(target_variables)

    if legacy_W.shape[0] != 2 * V:
        raise ValueError(
            f"Legacy final layer {weight_key!r} has first dim "
            f"{legacy_W.shape[0]}, expected {2 * V}."
        )
    if legacy_b.shape[0] != 2 * V:
        raise ValueError(
            f"Legacy final layer {bias_key!r} has shape {legacy_b.shape}, "
            f"expected first dim {2 * V}."
        )

    new_state: dict = {}
    # Pass through every key except the two we're splitting.
    for k, v in legacy_state.items():
        if k in (weight_key, bias_key):
            continue
        new_state[k] = v

    # Split into V chunks of [2, H] and [2]; key under each var name.
    # Row order is (mean_i, log_var_i) — preserved from the legacy code,
    # which packed predictions[2*i] = mean, predictions[2*i+1] = log_var.
    for i, var in enumerate(target_variables):
        new_state[f"heads.heads.{var}.linear.weight"] = legacy_W[2 * i : 2 * i + 2].clone()
        new_state[f"heads.heads.{var}.linear.bias"]   = legacy_b[2 * i : 2 * i + 2].clone()

    return new_state, weight_key, bias_key


def round_trip_check(
    legacy_state: dict,
    new_state: dict,
    target_variables: list[str],
    legacy_weight_key: str,
    legacy_bias_key: str,
) -> None:
    """Verify the migration was bit-for-bit lossless.

    Reconstruct what a single ``Linear(H, 2V)`` would have weighed by
    concatenating the heads' per-variable Linears in target_variable
    order, and assert torch.equal vs the original legacy tensor.

    Also confirms every other parameter passed through unchanged.

    Raises ``RuntimeError`` on any mismatch — caller should NOT proceed
    with the on-disk rename if this raises.
    """
    legacy_W = legacy_state[legacy_weight_key]
    legacy_b = legacy_state[legacy_bias_key]

    # Reconstruct legacy tensors from migrated chunks.
    rebuilt_W = torch.cat(
        [new_state[f"heads.heads.{var}.linear.weight"] for var in target_variables],
        dim=0,
    )
    rebuilt_b = torch.cat(
        [new_state[f"heads.heads.{var}.linear.bias"] for var in target_variables],
        dim=0,
    )
    if not torch.equal(rebuilt_W, legacy_W):
        max_diff = (rebuilt_W - legacy_W).abs().max().item()
        raise RuntimeError(
            f"Round-trip check FAILED on weight tensor: "
            f"reconstructed != legacy (max abs diff {max_diff:.3e}). "
            f"Migration would change model behaviour — refusing to write."
        )
    if not torch.equal(rebuilt_b, legacy_b):
        max_diff = (rebuilt_b - legacy_b).abs().max().item()
        raise RuntimeError(
            f"Round-trip check FAILED on bias tensor: "
            f"reconstructed != legacy (max abs diff {max_diff:.3e}). "
            f"Migration would change model behaviour — refusing to write."
        )

    # Body parameters: pass through unchanged.
    for k, v in legacy_state.items():
        if k in (legacy_weight_key, legacy_bias_key):
            continue
        if k not in new_state:
            raise RuntimeError(
                f"Round-trip check FAILED: body parameter {k!r} was "
                f"dropped during migration."
            )
        if not torch.equal(new_state[k], v):
            max_diff = (new_state[k] - v).abs().max().item()
            raise RuntimeError(
                f"Round-trip check FAILED: body parameter {k!r} unexpectedly "
                f"changed (max abs diff {max_diff:.3e})."
            )

    # New keys: only the heads. No surprise extras.
    expected_new_keys = {
        f"heads.heads.{var}.linear.weight" for var in target_variables
    } | {
        f"heads.heads.{var}.linear.bias" for var in target_variables
    }
    actual_new_keys = set(new_state) - set(legacy_state) | {
        legacy_weight_key, legacy_bias_key,
    } - {legacy_weight_key, legacy_bias_key}
    actual_new_keys = set(new_state) - (set(legacy_state) - {legacy_weight_key, legacy_bias_key})
    if actual_new_keys != expected_new_keys:
        raise RuntimeError(
            f"Round-trip check FAILED: unexpected new keys after migration. "
            f"expected={sorted(expected_new_keys)}, "
            f"actual_new={sorted(actual_new_keys)}."
        )


# ---------------------------------------------------------------------------
# Per-checkpoint workflow
# ---------------------------------------------------------------------------

def migrate_checkpoint_file(
    ckpt_path: Path,
    target_variables: list[str],
    tessera_injection: str,
    dry_run: bool,
) -> str:
    """Migrate one checkpoint .pt file. Returns a status string for logging."""
    legacy_backup = ckpt_path.with_suffix(".legacy.pt")
    if legacy_backup.exists():
        return f"SKIP (legacy backup already exists at {legacy_backup.name})"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    if _is_already_migrated(state):
        return "SKIP (already migrated — has heads.heads.* keys)"

    new_state, w_key, b_key = migrate_state_dict(
        state, target_variables, tessera_injection,
    )
    round_trip_check(state, new_state, target_variables, w_key, b_key)

    if dry_run:
        return f"DRY-RUN OK (would split {w_key} → {len(target_variables)} heads)"

    new_ckpt = dict(ckpt)
    new_ckpt["model_state_dict"] = new_state
    # Write migrated to a sibling .tmp file, then rename — never leave a
    # half-written file in the run dir.
    tmp_path = ckpt_path.with_suffix(".tmp.pt")
    torch.save(new_ckpt, tmp_path)
    # Backup original.
    shutil.move(str(ckpt_path), str(legacy_backup))
    # Move temp into place.
    shutil.move(str(tmp_path), str(ckpt_path))
    return f"OK (split {w_key} into {len(target_variables)} heads; backup: {legacy_backup.name})"


def migrate_run_dir(run_dir: Path, dry_run: bool) -> None:
    """Migrate one training-run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.info(f"  {run_dir.name}: skip — no config.json")
        return
    config = json.loads(config_path.read_text())

    target_variables = config.get("target_variables") or [
        config.get("target_variable", "tmax")
    ]
    tessera_injection = config.get("tessera_injection", "concat")

    if tessera_injection == "hypernet":
        logger.warning(
            f"  {run_dir.name}: REFUSE — tessera_injection=hypernet, "
            "no migration possible. Original files left untouched."
        )
        return

    logger.info(
        f"  {run_dir.name}: target_variables={target_variables}, "
        f"injection={tessera_injection}"
    )

    # Migrate each .pt file in the run dir.
    for stem in ("best_model", "latest_model"):
        ckpt_path = run_dir / f"{stem}.pt"
        if not ckpt_path.exists():
            continue
        try:
            status = migrate_checkpoint_file(
                ckpt_path, target_variables, tessera_injection, dry_run,
            )
            logger.info(f"    {stem}.pt: {status}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"    {stem}.pt: FAILED — {e}")
            return  # bail out on this run dir; don't touch config.

    # Augment config.json with likelihood_per_variable (always all-Gaussian
    # for legacy checkpoints) — and back up the original.
    if "likelihood_per_variable" not in config:
        legacy_config_backup = run_dir / "config.legacy.json"
        if dry_run:
            logger.info(
                f"    config.json: DRY-RUN would add "
                f"likelihood_per_variable: all-gaussian"
            )
        elif legacy_config_backup.exists():
            logger.info(
                "    config.json: legacy backup already exists; skipping config update"
            )
        else:
            shutil.copy(str(config_path), str(legacy_config_backup))
            config["likelihood_per_variable"] = {
                var: "gaussian" for var in target_variables
            }
            config_path.write_text(json.dumps(config, indent=2))
            logger.info(
                f"    config.json: added likelihood_per_variable=all-gaussian; "
                f"backup: {legacy_config_backup.name}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy ConvCNP checkpoints to the v4 per-variable "
            "heads layout. Idempotent and rollback-safe (originals are "
            "kept as best_model.legacy.pt / config.legacy.json)."
        ),
    )
    parser.add_argument(
        "root", type=Path,
        help="Root directory containing training-run subdirectories. "
             "Each subdir with a config.json and best_model.pt is migrated.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be migrated without writing anything.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress info logging; only warnings + errors.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    if not args.root.is_dir():
        logger.error(f"Not a directory: {args.root}")
        sys.exit(1)

    logger.info(f"Migrating training runs under {args.root} "
                f"(dry-run={args.dry_run})")

    # A "run dir" has a config.json and at least one *.pt file.
    run_dirs = sorted(
        d for d in args.root.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )
    if not run_dirs:
        # Maybe the root itself is a run dir.
        if (args.root / "config.json").exists():
            run_dirs = [args.root]
        else:
            logger.error(
                f"No run directories found under {args.root}. "
                "Either pass a single run dir or a parent of run dirs."
            )
            sys.exit(1)

    logger.info(f"Found {len(run_dirs)} candidate run dir(s).")

    for run_dir in run_dirs:
        migrate_run_dir(run_dir, args.dry_run)

    logger.info("Done.")


if __name__ == "__main__":
    main()