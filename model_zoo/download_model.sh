#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(dirname -- "$0")
cd "${SCRIPT_DIR}" || exit 1

INPUT_FILE="./urls.txt"

if [[ ! -f "${INPUT_FILE}" ]]; then
	echo "[ERROR] Missing ${INPUT_FILE}"
	exit 1
fi

download_entry() {
	local url="$1"
	local dir="$2"
	local out="$3"
	local auto_rename="$4"

	mkdir -p "${dir}"

	# If output name is not provided, derive from URL path (without query string).
	if [[ -z "${out}" ]]; then
		local no_query="${url%%\?*}"
		out="${no_query##*/}"
	fi

	local dest="${dir}/${out}"

	if [[ "${auto_rename}" == "false" ]]; then
		wget -nv -O "${dest}" "${url}"
	else
		# Keep default wget behavior when auto-renaming is enabled.
		(cd "${dir}" && wget -nv "${url}")
	fi
}

current_url=""
current_dir="."
current_out=""
current_auto="false"

while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
	line="${raw_line%$'\r'}"

	# Skip blank lines and comments.
	if [[ -z "${line//[[:space:]]/}" ]] || [[ "${line}" =~ ^[[:space:]]*# ]]; then
		continue
	fi

	if [[ "${line}" =~ ^https?:// ]]; then
		# Flush previous entry.
		if [[ -n "${current_url}" ]]; then
			download_entry "${current_url}" "${current_dir}" "${current_out}" "${current_auto}"
		fi

		current_url="${line}"
		current_dir="."
		current_out=""
		current_auto="false"
		continue
	fi

	# Parse indented key=value options following a URL.
	opt="${line#${line%%[![:space:]]*}}"
	key="${opt%%=*}"
	value="${opt#*=}"

	case "${key}" in
		dir)
			current_dir="${value}"
			;;
		out)
			current_out="${value}"
			;;
		auto-file-renaming)
			current_auto="${value}"
			;;
	esac
done < "${INPUT_FILE}"

# Flush final entry.
if [[ -n "${current_url}" ]]; then
	download_entry "${current_url}" "${current_dir}" "${current_out}" "${current_auto}"
fi

echo "[INFO] Download complete."
