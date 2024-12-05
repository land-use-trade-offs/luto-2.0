#!/bin/bash

# Read the settings_bash file ==> JOBNAME, QUEUE, NCPUS, MEM, NCPUS, QUEUE
source luto/settings_bash.py

# Activate the Conda environment and get the path to the Python executable
source ~/.bashrc
conda activate luto
PYTHON=$(which python)

# Create a temporary script file
SCRIPT_PBS=$(mktemp)

echo '''
import luto.simulation as sim
import luto.settings as settings
data = sim.load_data_from_disk(f"{settings.INPUT_DIR}/Data_RES{settings.RESFACTOR}.pkl")
sim.run(data=data, base=2010, target=2050)
''' > python_script.py


cat << EOF > $SCRIPT_PBS
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l storage=scratch/${PROJECT}+gdata/${PROJECT}
#PBS -l ncpus=${NCPUS}
#PBS -l mem=${MEM}
#PBS -l jobfs=10GB
#PBS -l walltime=48:00:00
#PBS -l wd="$(dirname "$0")"

${PYTHON} python_script.py
EOF

# Submit the job to PBS
qsub ${SCRIPT_PBS}

# Remove the temporary script file
rm $SCRIPT_PBS