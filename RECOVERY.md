# Recovery procedures

Concrete recovery procedures and known repro/fix pairs for incidents seen on
the H200 box during Phase 0–2 setup of this repo. Maintained as the box
brings up new failure modes.

## 1. DLAMI driver/fabricmanager version mismatch (CUDA Error 802)

**Affected AMI:** `ami-0bae40837d7422a24` (Deep Learning OSS Nvidia Driver AMI
GPU PyTorch 2.11 (Ubuntu 24.04) 20260517, `us-east-2`).
**Hardware:** AWS `p5en.48xlarge` (HGX H200 with NVSwitch).
**Date observed:** 2026-05-20.

### Symptom

`torch.cuda.is_available()` returns `False` even though `nvidia-smi -L`
shows 8 GPUs. Direct CUDA call fails with:

```
RuntimeError: Unexpected error from cudaGetDeviceCount().
Error 802: system not yet initialized
```

NCCL `init_process_group` cannot proceed; multi-rank calibration is blocked.

### Root cause

DLAMI bake-time version skew. The AMI ships with:

| Component | Version |
|---|---|
| `/proc/driver/nvidia/version` (loaded kernel module) | **595.64** |
| `nvidia-kernel-common-595-server` (userspace firmware/headers) | **595.71.05** |
| `nvidia-fabricmanager-595` (only version in apt repo) | **595.71.05** |

`nv-fabricmanager` checks the loaded driver interface version and refuses
to start when it doesn't match exactly:

```
nv-fabricmanager: fabric manager NVIDIA GPU driver interface version
595.71.05 don't match with driver version 595.64. Please update with
matching NVIDIA driver package.
```

On HGX H200 (NVSwitch-equipped) boxes, fabric-manager is required to
initialize the NVSwitch routing fabric. Without it, CUDA can enumerate
GPUs but cannot complete `cudaInitDevice` → Error 802.

### Fix

```bash
# Install the precompiled 595.71.05 kernel modules + DKMS source so
# modules.dep points at the correct .ko after reboot:
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    nvidia-dkms-595-server \
    linux-modules-nvidia-595-server-aws-6.17

# The DKMS install may report errors trying to overwrite the precompiled
# modules — that's fine, the precompiled ones in
# /lib/modules/<kernel>/kernel/nvidia-595srv/ are what we want.

# OS-level reboot (instance store is preserved across `sudo reboot`,
# unlike `aws ec2 stop` which wipes /opt/dlami/nvme):
sudo reboot

# After reboot — verify:
cat /proc/driver/nvidia/version  # → expect 595.71.05
systemctl status nvidia-fabricmanager.service  # → expect active (running)
nvidia-smi -L  # → 8 GPUs
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
#   → True 8

# If nvidia-smi was removed during install dance, reinstall:
sudo apt-get install -y nvidia-utils-595-server libnvidia-compute-595-server
```

### Side-effects to watch for

After the apt install, `dpkg -l linux-image-6.17.0-1015-aws` may report
post-install hook errors trying to re-run DKMS for nvidia-srv. This is
harmless — the precompiled modules under `kernel/nvidia-595srv/` are the
ones loaded after reboot. The hook just fails because there's no DKMS
source for 595.71.05 (only the precompiled .ko). To silence on next kernel
upgrade, remove the stale DKMS entry: `sudo dkms remove nvidia/595.64 --all`
(may report "not in DKMS tree" if already gone — harmless).

### Rollback procedure (if 595.71.05 modules fail to load)

The original 595.64 modules and userspace are backed up at
`/root/nvidia-modules-595.71.05.bak/`:

```
/root/nvidia-modules-595.71.05.bak/
├── kernel-modules/       # /lib/modules/<kernel>/kernel/nvidia-595srv/* snapshot (543 MB)
└── share-nvidia/         # /usr/share/nvidia snapshot (12 MB)
```

If a reboot leaves the box wedged without working CUDA, recovery steps
(from the AWS console or via `aws ec2 send-diagnostic-interrupt`):

1. Boot single-user mode, mount root rw.
2. Restore the 595.64 modules into place:
   ```bash
   rm -rf /lib/modules/6.17.0-1015-aws/kernel/nvidia-595srv
   cp -a /root/nvidia-modules-595.71.05.bak/kernel-modules \
         /lib/modules/6.17.0-1015-aws/kernel/nvidia-595srv
   depmod -a 6.17.0-1015-aws
   ```
3. Reboot back into multi-user. fabric-manager will FAIL to start on the
   downgraded driver — at this point you have GPUs visible to `nvidia-smi`
   but no NCCL, equivalent to the original DLAMI state. Open AWS Support
   if a fresh AMI is needed.

### Upstream PR / bug to file

- **AWS DLAMI**: file a ticket against `aws/deep-learning-amis` GitHub
  repo and an AWS Support case. Title pattern: `DLAMI <ami-id> ships
  driver/fabricmanager version mismatch; CUDA Error 802 on p5en/p6
  instances out of the box`. Include this section as the repro.
- See `CONTRIBUTIONS_QUEUE.md` for status.

## 2. `mtp.*` keys silently dropped by transformers 5.8.1

See `patches/modeling_deepseek_v4.py.diff` and
`patches/VERSIONS.md` — `DeepseekV4PreTrainedModel` has
`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` which causes
`from_pretrained` to drop every `mtp.*` weight silently. The patch
neutralizes that regex.

Already documented in `CLAUDE.md` and the cross-project memory at
`~/.claude/projects/-home-paul/memory/dsv4_silent_mtp_drop.md`. Upstream
PR candidacy: yes, against `huggingface/transformers`. See
`CONTRIBUTIONS_QUEUE.md`.

## 3. Dequant `--device cuda` fails with Error 802 on first call

If CUDA Error 802 returns on a system that previously worked, `nvidia-smi`
is the canonical "warm the GPU subsystem up" first call. The dequant
script also accepts `--device cpu` as a safe fallback; the 543 GB BF16
output completes in ~17 minutes either way (workload is IO/RAM bound).
