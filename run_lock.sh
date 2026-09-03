#!/bin/bash
source /mnt/scratch/bgxp240/interp/activate.sh >/dev/null 2>&1
cd /mnt/scratch/bgxp240/interp/evalaware
python -c "from evalaware import lock_metrics; lock_metrics.lock()"
echo "LOCK DONE rc=$?"
