"""Verify the ROCm torch build on an MI355X node: device visible, the wheel
carries kernels for the node's GPU architecture, a matmul runs, and the small
Aurora model executes one rollout step on random 0.25 deg input (exercises the
Swin attention / patch-embedding kernels the real extraction uses).

Run by ``setup_csd3_env.sh --gpu-check`` inside an interactive GPU job; needs
the checkpoints prefetched into ``HF_HOME`` (the setup script does that).
"""

from __future__ import annotations

import datetime as dt
import sys
import time

import torch

print(f"torch {torch.__version__}  hip {torch.version.hip}")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False -- no GPU visible to the ROCm build")
props = torch.cuda.get_device_properties(0)
arch = getattr(props, "gcnArchName", "?")
arch_list = torch.cuda.get_arch_list()
print(f"device 0: {props.name}  arch {arch}  mem {props.total_memory / 2**30:.0f} GiB")
print(f"wheel arch list: {arch_list}")
base = arch.split(":")[0]
if arch_list and not any(a.startswith(base) for a in arch_list):
    sys.exit(
        f"the torch wheel has no kernels for {base}; rebuild the env with a newer "
        "TORCH_BACKEND / TORCH_OVERRIDES (see scripts/aurora/setup_csd3_env.sh)"
    )

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
y = x @ x
torch.cuda.synchronize()
err = float((y[:8, :8].cpu() - (x[:8].cpu() @ x.cpu())[:, :8]).abs().max())
print(f"matmul 4096^2: {time.time() - t0:.3f} s, max abs err vs cpu {err:.3g}")

from aurora import AuroraSmallPretrained, Batch, Metadata, rollout  # noqa: E402

model = AuroraSmallPretrained()
model.load_checkpoint()
model = model.eval().to("cuda")
H, W, L = 721, 1440, 13
batch = Batch(
    surf_vars={k: torch.randn(1, 2, H, W) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(H, W) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, L, H, W) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, H),
        lon=torch.linspace(0, 360, W + 1)[:-1],
        time=(dt.datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
    ),
).to("cuda")
torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    preds = list(rollout(model, batch, steps=2))
torch.cuda.synchronize()
dt_s = time.time() - t0
out = preds[-1].surf_vars["2t"]
print(
    f"AuroraSmall rollout 2 steps on random input: {dt_s:.2f} s, "
    f"2t shape {tuple(out.shape)}, finite={bool(torch.isfinite(out).all())}"
)
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print("GPU CHECK OK")
