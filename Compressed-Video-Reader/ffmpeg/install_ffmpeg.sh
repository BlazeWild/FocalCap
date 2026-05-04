#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname -- "$0")")
cd "${SCRIPT_DIR}" || exit 1

log() {
  echo "[cv-reader][ffmpeg] $*"
}

die() {
  echo "[cv-reader][ffmpeg][error] $*" 1>&2
  exit 1
}

# No-sudo by default. Enable with: CVR_USE_SUDO=1
CVR_USE_SUDO=${CVR_USE_SUDO:-0}

# Default install locations (user-local)
HOME_DIR=${HOME:-"${SCRIPT_DIR}"}
FFMPEG_PREFIX=${FFMPEG_PREFIX:-"${HOME_DIR}/.local/cvreader-ffmpeg"}
FFMPEG_BINDIR=${FFMPEG_BINDIR:-"${HOME_DIR}/.local/bin"}

if [[ "${CVR_USE_SUDO}" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing FFmpeg build dependencies via apt-get (sudo enabled)"
    sudo apt-get update
    sudo apt-get install -y \
      autoconf \
      automake \
      build-essential \
      cmake \
      git \
      libass-dev \
      libfreetype6-dev \
      libtool \
      libva-dev \
      libvdpau-dev \
      libvorbis-dev \
      libxcb1-dev \
      libxcb-shm0-dev \
      libxcb-xfixes0-dev \
      pkg-config \
      texinfo \
      wget \
      zlib1g-dev \
      nasm \
      yasm \
      libx264-dev \
      libx265-dev \
      libnuma-dev \
      libvpx-dev \
      libmp3lame-dev \
      libopus-dev
  else
    die "CVR_USE_SUDO=1 requested, but apt-get was not found on this system. Install build deps manually."
  fi
else
  log "Skipping system dependency install (no sudo)."
  log "If configure fails, install build deps yourself or rerun with CVR_USE_SUDO=1."
fi

# Download FFMPEG source
FFMPEG_VERSION=${FFMPEG_VERSION:-"5.1"}
log "Downloading FFmpeg source (${FFMPEG_VERSION})"
mkdir -p "${SCRIPT_DIR}/ffmpeg_source"
rm -f "${SCRIPT_DIR}/ffmpeg-snapshot.tar.bz2"
wget -O "${SCRIPT_DIR}/ffmpeg-snapshot.tar.bz2" "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2"
rm -rf "${SCRIPT_DIR}/ffmpeg_source"/*
tar xjvf "${SCRIPT_DIR}/ffmpeg-snapshot.tar.bz2" -C "${SCRIPT_DIR}/ffmpeg_source" --strip-components=1
rm -f "${SCRIPT_DIR}/ffmpeg-snapshot.tar.bz2"

# Install patch for FFMPEG
log "Patching FFmpeg"
export FFMPEG_INSTALL_DIR="${SCRIPT_DIR}/ffmpeg_source"
export FFMPEG_PATCH_DIR="${SCRIPT_DIR}/ffmpeg_patch"

chmod +x "$FFMPEG_PATCH_DIR"/patch.sh
"$FFMPEG_PATCH_DIR"/patch.sh || exit 1

log "Configuring FFmpeg"
cd "${SCRIPT_DIR}/ffmpeg_source" || exit 1
chmod +x ./configure

mkdir -p "${FFMPEG_PREFIX}" "${FFMPEG_BINDIR}"

# Build ffprobe and ffmpeg by default (ffplay disabled). Avoid fully-static builds to reduce dependency pain.
# If yasm is not installed, fall back to --disable-yasm for compatibility.
YASM_FLAG=""
if ! command -v yasm >/dev/null 2>&1; then
  YASM_FLAG="--disable-yasm"
fi

# Some toolchain/arch combos hit assembler errors in FFmpeg's x86 inline asm
# (e.g., "operand type mismatch for `shr'" from libavcodec/x86/mathops.h).
# Default to disabling asm to make builds more portable; you can opt out.
FFMPEG_DISABLE_ASM=${FFMPEG_DISABLE_ASM:-1}
ASM_FLAGS=()
if [[ "${FFMPEG_DISABLE_ASM}" == "1" ]]; then
  ASM_FLAGS+=(--disable-asm --disable-inline-asm --disable-x86asm)
  log "FFMPEG_DISABLE_ASM=1: disabling asm/inline-asm/x86asm for portability"
fi

CONFIGURE_FLAGS=(
  --prefix="${FFMPEG_PREFIX}"
  --bindir="${FFMPEG_BINDIR}"
  --enable-pic
  --disable-doc
  --disable-debug
  --disable-ffplay
)

if [[ -n "${YASM_FLAG}" ]]; then
  CONFIGURE_FLAGS+=("${YASM_FLAG}")
fi

CONFIGURE_FLAGS+=("${ASM_FLAGS[@]}")

./configure "${CONFIGURE_FLAGS[@]}"

log "Compiling FFmpeg"
JOBS=${JOBS:-"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"}
make -j"${JOBS}"
make install

if [[ ! -x "${FFMPEG_BINDIR}/ffprobe" ]]; then
  die "ffprobe was not installed at ${FFMPEG_BINDIR}/ffprobe"
fi

log "Installed ffprobe: $(${FFMPEG_BINDIR}/ffprobe -version | head -n 1)"

if ! command -v ffprobe >/dev/null 2>&1; then
  log "ffprobe is installed but not on your PATH."
  log "Add this to your shell rc (e.g. ~/.bashrc):"
  log "  export PATH=\"${FFMPEG_BINDIR}:\$PATH\""
fi
