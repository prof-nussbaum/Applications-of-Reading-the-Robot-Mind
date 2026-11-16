#!/bin/bash
# Q3 driver: extraction then scoring; monitor.log records DONE/FAILED.
cd /home/ecpi-student-05/robot-mind/q3-pick-tracking
PY=/home/ecpi-student-05/robot-mind/venv/bin/python
echo "$(date -Is) START" >> monitor.log
if $PY run_q3.py >> q3_run.log 2>&1 && $PY score_q3.py >> q3_run.log 2>&1; then
  echo "$(date -Is) DONE" >> monitor.log
else
  echo "$(date -Is) FAILED rc=$?" >> monitor.log
fi
