#!/usr/bin/env bash
# One-command setup inside the AMD hackathon JupyterLab session (notebooks.amd.com/hackathon).
#
#   bash scripts/notebook_setup.sh
#
# pip installs inside a session vanish when the daily quota resets, so dependencies go into persistent
# storage once and are re-used via PYTHONPATH.
set -euo pipefail

# Persistent storage is /persistent on "jupyter-hack-*" pods and /workspace on "rgapi-hackathon-*" pods.
if [ -d /persistent ] && [ -w /persistent ]; then PERSIST=/persistent; else PERSIST=/workspace; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$PERSIST/pydeps-swecrew"
mkdir -p "$DEPS" "$PERSIST/hf-cache"

export PYTHONPATH="$DEPS:$REPO/academy-core/src:$REPO/src:${PYTHONPATH:-}"
export HF_HOME="$PERSIST/hf-cache"

if [ ! -f "$DEPS/.installed" ]; then
  python3 -m pip install --target "$DEPS" -r "$REPO/requirements.txt" pytest pytest-asyncio
  touch "$DEPS/.installed"
fi

cat > "$PERSIST/env-swecrew.sh" <<EOF
export PYTHONPATH="$PYTHONPATH"
export HF_HOME="$HF_HOME"
export PERSIST="$PERSIST"
EOF

python3 - <<'PY'
import torch
ok = torch.cuda.is_available()
print("ROCm GPU available:", ok)
if ok:
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM total {total/2**30:.1f} GiB, free {free/2**30:.1f} GiB")
PY
echo "Done. In new terminals run: source $PERSIST/env-swecrew.sh"
echo "Remember: the 3 h/day quota counts idle time - press 'Turn-off Session' when you stop."
