#!/usr/bin/env bash
set -euo pipefail

# Optional: create a Python 3.10 environment (the repo README's recommended version)
# without touching system Python. Installs micromamba locally into the repo folder.

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-.micromamba}"
MAMBA_BIN_DIR="${MAMBA_BIN_DIR:-.local/bin}"
MAMBA_BIN="${MAMBA_BIN_DIR}/micromamba"
ENV_NAME="${ENV_NAME:-llmopt-py310}"

mkdir -p "${MAMBA_BIN_DIR}"

if [ ! -x "${MAMBA_BIN}" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download micromamba" >&2
    exit 1
  fi
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  # micromamba official download endpoint (linux-64 tar.bz2 containing bin/micromamba)
  curl -LfsS "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "${tmpdir}/micromamba.tar.bz2"
  tar -xjf "${tmpdir}/micromamba.tar.bz2" -C "${tmpdir}"
  install -m 0755 "${tmpdir}/bin/micromamba" "${MAMBA_BIN}"
fi

export MAMBA_ROOT_PREFIX

eval "$("${MAMBA_BIN}" shell hook -s bash)"

if ! "${MAMBA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${MAMBA_BIN}" create -y -n "${ENV_NAME}" python=3.10 pip
fi

micromamba activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

cat <<EOF
Done.

Activate later with:
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}"
  eval "$(${MAMBA_BIN} shell hook -s bash)"
  micromamba activate ${ENV_NAME}
EOF

