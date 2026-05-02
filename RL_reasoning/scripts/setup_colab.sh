#!/usr/bin/env bash
# Usage:
#   !bash /content/omscs/scripts/setup_colab.sh
#
set -euo pipefail

WORKDIR=/content/work
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1. Clone Open-RS upstream.
if [ ! -d open-rs ]; then
  git clone https://github.com/knoveleng/open-rs.git
fi

# 2. Refresh pip.
pip install -q --upgrade pip

# 3. Install Open-RS in editable mode WITH its own pinned deps + eval extras
pip install -q -e ./open-rs[eval,code]

# 4. vLLM (not declared in Open-RS install_requires).
pip install -q "vllm==0.7.3"

# 4b. Pin trl to 0.15.2 — newer 0.16.0.dev0 from open-rs's deps removes
# attributes that open-rs's own utils still reference. Force install.
pip install -q "trl==0.15.2" --force-reinstall --no-deps

# 4c. bitsandbytes for paged 8-bit AdamW (needed to fit 1.5B + ref + vLLM
# rollout cache on a single A100-40GB).
pip install -q bitsandbytes

# 4d. protobuf >= 5.27 — TensorFlow (pre-installed on Colab and pulled in by
pip install -q "protobuf>=5.27,<6"

# 5. flash-attn — let pip pick the version that matches Colab's torch wheel.
pip install -q flash-attn --no-build-isolation || \
  echo "WARN: flash-attn install failed. Set attn_implementation: sdpa in the config."

echo "Setup complete"
