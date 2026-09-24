#!/usr/bin/env bash
# Pre-submission gate check: runs the same checks as the official grader, locally.
#
#   scripts/check_submission.sh <image:tag> <samples_dir> [per_item_timeout_s]
#
# Checks: (1) final image built on the mandated ROCm base (layer identity), (2) size <= 60 GiB,
# (3) container reaches ready within 10 min, (4) every sample runs within the per-item limit and writes
# /app/output/<stem>_output.json, (5) peak VRAM within [1, 48.48] GiB, (6) no obvious secrets in the image.
set -euo pipefail

IMAGE="${1:?usage: check_submission.sh <image:tag> <samples_dir> [per_item_timeout_s]}"
SAMPLES="${2:?samples dir required}"
ITEM_TIMEOUT="${3:-30}"
BASE="rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
NAME="submission-check-$$"
fail() { echo "FAIL: $*"; docker rm -f "$NAME" >/dev/null 2>&1 || true; exit 1; }
pass() { echo "PASS: $*"; }

echo "== 1. base image layer identity"
docker pull -q "$BASE" >/dev/null
base_layers=$(docker image inspect "$BASE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')
image_layers=$(docker image inspect "$IMAGE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')
n=$(printf '%s\n' "$base_layers" | grep -c . || true)
[ "$(printf '%s\n' "$image_layers" | head -n "$n")" = "$base_layers" ] || fail "lower layers differ from $BASE (squashed or wrong base?)"
pass "image is layered on the mandated base ($n layers)"

echo "== 2. uncompressed size"
size=$(docker image inspect "$IMAGE" --format '{{.Size}}')
limit=$((60 * 1024 * 1024 * 1024))
[ "$size" -le "$limit" ] || fail "image is $((size / 1024 / 1024 / 1024)) GiB > 60 GiB"
pass "size $((size / 1024 / 1024)) MiB"

echo "== 3. secrets scan"
if docker run --rm --entrypoint sh "$IMAGE" -c 'ls -a /app; find / -xdev \( -name ".env" -o -name "*.pem" -o -name "id_rsa" \) 2>/dev/null | grep -v -E "^/(proc|sys|usr/lib|usr/share|etc/ssl)" | head' | grep -E "\.env|\.pem|id_rsa"; then
  fail "possible secret files found in image"
fi
pass "no secret files found"

echo "== 4. startup within 10 minutes"
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size 16g -v "$(realpath "$SAMPLES")":/app/input:ro "$IMAGE" >/dev/null
start=$(date +%s)
until docker exec "$NAME" test -f /tmp/academy_ready 2>/dev/null; do
  [ $(( $(date +%s) - start )) -lt 600 ] || fail "container not ready after 600 s"
  sleep 5
done
pass "ready after $(( $(date +%s) - start )) s"

echo "== 5. per-item runs + VRAM sampling"
peak=0
( while docker inspect "$NAME" >/dev/null 2>&1; do
    v=$( (amd-smi metric --mem --json 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);d=d if isinstance(d,list) else [d];print(sum(int(float((x.get("mem_usage") or {}).get("used_vram",{}).get("value",0))) for x in d))') || echo 0)
    echo "$v" >> /tmp/$NAME.vram; sleep 3
  done ) &
sampler=$!
for f in "$SAMPLES"/*; do
  stem=$(basename "${f%.*}")
  t0=$(date +%s.%N)
  timeout "$ITEM_TIMEOUT" docker exec "$NAME" python3 /app/app.py --input "/app/input/$(basename "$f")" >/dev/null \
    || fail "$stem exceeded ${ITEM_TIMEOUT}s or crashed"
  dt=$(echo "$(date +%s.%N) - $t0" | bc)
  docker exec "$NAME" test -f "/app/output/${stem}_output.json" || fail "missing /app/output/${stem}_output.json"
  echo "  $stem: ${dt}s -> $(docker exec "$NAME" cat "/app/output/${stem}_output.json" | tr -d '\n' | cut -c1-120)"
done
kill "$sampler" 2>/dev/null || true
peak=$(sort -n /tmp/$NAME.vram 2>/dev/null | tail -1 || echo 0)
rm -f /tmp/$NAME.vram
echo "  peak VRAM (MB, from amd-smi): ${peak:-unknown}"
if [ -n "$peak" ] && [ "$peak" != "0" ]; then
  [ "$peak" -ge 1024 ] || fail "peak VRAM ${peak} MB < 1 GiB (the GPU must be used)"
  [ "$peak" -le 49643 ] || fail "peak VRAM ${peak} MB > 48 GiB + 1%"
  pass "VRAM within limits"
else
  echo "WARN: could not read VRAM (amd-smi missing?) - verify manually: watch -n1 'amd-smi metric --mem'"
fi

docker rm -f "$NAME" >/dev/null
echo "ALL CHECKS PASSED for $IMAGE"
