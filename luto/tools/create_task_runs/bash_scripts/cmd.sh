#!/bin/bash

# Read task_param.py ==> JOB_NAME, QUEUE, NCPUS, MEM, TIME
source task_param.py

# If redo_param.py exists (resubmission of a checkpoint run, possibly with
# overridden resources/job name), source it to override the values above.
if [ -f redo_param.py ]; then
    source redo_param.py
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT="${PROJECT:-$(echo "${SCRIPT_DIR}" | grep -oP '(?<=/g/data/)[^/]+')}"

SCRIPT_PBS=$(mktemp)

cat << EOF > $SCRIPT_PBS
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l storage=scratch/${PROJECT}+gdata/${PROJECT}
#PBS -l ncpus=${NCPUS}
#PBS -l mem=${MEM}
#PBS -l jobfs=100GB
#PBS -l walltime=${TIME}
#PBS -l wd

export JOBLIB_TEMP_FOLDER=\$PBS_JOBFS

conda run -n luto python "${SCRIPT_DIR}/python_script.py"
EOF

qsub "${SCRIPT_PBS}"
rm "${SCRIPT_PBS}"
