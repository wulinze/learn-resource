#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_FILE="${SCRIPT_DIR}/test_swiglu_forward_and_per_token_cast.py"
REL_TEST_FILE="${TEST_FILE#${REPO_ROOT}/}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-ncu-reports/puzzles/l06_fusion/swiglu_forward_and_per_token_cast}"
NCU_SET="${NCU_SET:-basic}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:swiglu_forward_and_per_token_cast}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
NCU_PRINT_SUMMARY="${NCU_PRINT_SUMMARY:-per-kernel}"
DRY_RUN="${DRY_RUN:-0}"

sanitize_report_name() {
  sed -E 's/[^A-Za-z0-9_.-]+/-/g; s/^-+//; s/-+$//' | cut -c1-180
}

cd "${REPO_ROOT}" || exit 1

mapfile -t NODEIDS < <("${PYTHON_BIN}" -m pytest --collect-only -q "${REL_TEST_FILE}" "$@" | awk '/::/ {print}')
if [[ "${#NODEIDS[@]}" -eq 0 ]]; then
  echo "No pytest node IDs collected from ${REL_TEST_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
failures=()

echo "Collected ${#NODEIDS[@]} test cases from ${REL_TEST_FILE}"
for nodeid in "${NODEIDS[@]}"; do
  report_name="$(printf '%s' "${nodeid}" | sanitize_report_name)"
  cmd=(
    "${PYTHON_BIN}" -m pytest
    -p tests.pytest_benchmark_plugin
    "${nodeid}"
    --ncu-profile
    --ncu-output-dir "${OUTPUT_DIR}"
    --ncu-output-name "${report_name}"
    --ncu-set "${NCU_SET}"
    --ncu-launch-count "${NCU_LAUNCH_COUNT}"
    --ncu-target-processes "${NCU_TARGET_PROCESSES}"
    --ncu-print-summary "${NCU_PRINT_SUMMARY}"
    -q
  )
  if [[ -n "${NCU_PATH:-}" ]]; then
    cmd+=(--ncu-path "${NCU_PATH}")
  fi
  if [[ "${NCU_LAUNCH_SKIP}" != "0" ]]; then
    cmd+=(--ncu-launch-skip "${NCU_LAUNCH_SKIP}")
  fi
  if [[ -n "${NCU_KERNEL_NAME}" ]]; then
    cmd+=(--ncu-kernel-name "${NCU_KERNEL_NAME}")
  fi
  cmd+=("$@")

  echo
  echo "=== NCU: ${nodeid} ==="
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  "${cmd[@]}"
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    failures+=("${nodeid}: ${status}")
  fi
done

if [[ "${#failures[@]}" -ne 0 ]]; then
  echo
  echo "Failures:"
  printf '  %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "NCU reports written under ${OUTPUT_DIR}"
