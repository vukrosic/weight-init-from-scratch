#!/bin/sh
# Runs every step, the checks and the exercise, in the order of the video.
# The long trainings are read from files/runs.jsonl.
# To train the table again: delete that file, then  python3 sweep.py 0 1
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_network.py step2_same.py step3_deep.py step4_sum.py \
         step5_layer.py step6_back.py step7_dial.py step8_rules.py \
         step9_checks.py tests.py exercise.py; do
  echo
  echo "=== $f ==="
  $PY -W ignore $f
done
