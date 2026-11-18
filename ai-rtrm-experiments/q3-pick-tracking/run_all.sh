#!/bin/bash
# Q3 driver: extraction then scoring; monitor.log records DONE/FAILED.
ROBOT_MIND="${ROBOT_MIND:-/home/ecpi-student-05/robot-mind}"  # Jetson working tree; override with $ROBOT_MIND
cd "$ROBOT_MIND/q3-pick-tracking"
PY="$ROBOT_MIND/venv/bin/python"
echo "$(date -Is) START" >> monitor.log
if $PY run_q3.py >> q3_run.log 2>&1 && $PY score_q3.py >> q3_run.log 2>&1; then
  echo "$(date -Is) DONE" >> monitor.log
else
  echo "$(date -Is) FAILED rc=$?" >> monitor.log
fi
