#!/bin/bash
source /mnt/scratch/bgxp240/interp/activate.sh >/dev/null 2>&1
cd /mnt/scratch/bgxp240/interp/evalaware
python -u -c "from evalaware import sensitivity; sensitivity.report()"
echo "SENS DONE rc=$?"
