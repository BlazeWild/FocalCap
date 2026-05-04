#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname -- "$0")")
cd "${SCRIPT_DIR}" || exit 1

bash ffmpeg/install_ffmpeg.sh

# Ensure the ffprobe we just installed is usable for the rest of this script.
# Keep defaults in sync with ffmpeg/install_ffmpeg.sh.
HOME_DIR=${HOME:-""}
DEFAULT_FFMPEG_BINDIR="${HOME_DIR}/.local/bin"
FFMPEG_BINDIR=${FFMPEG_BINDIR:-"${DEFAULT_FFMPEG_BINDIR}"}
if [[ -d "${FFMPEG_BINDIR}" ]]; then
	export PATH="${FFMPEG_BINDIR}:${PATH}"
fi

if command -v ffprobe >/dev/null 2>&1; then
	echo "[cv-reader] ffprobe OK: $(ffprobe -version | head -n 1)"
else
	echo "[cv-reader][error] ffprobe is not on PATH (install did not succeed or PATH is misconfigured)." 1>&2
	if [[ -n "${HOME_DIR}" ]]; then
		echo "[cv-reader][error] If you used the default install, add this to your shell rc: export PATH=\"${HOME_DIR}/.local/bin:\$PATH\"" 1>&2
	fi
	exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
	python3 -m pip install .
else
	# No-sudo default for system Python installs.
	python3 -m pip install --user .
fi

# test if installation is successful
cv_reader -h || (echo "Installation failed!" && exit 1)
