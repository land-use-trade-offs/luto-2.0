#!/bin/bash

# Read the settings_bash file ==> JOBNAME, QUEUE, NCPUS, MEM, NCPUS, QUEUE
source luto/settings_bash.py

# Activate the Conda environment and get the path to the Python executable
source ~/.bashrc
conda activate luto
PYTHON=$(which python)

# Create a temporary script file
SCRIPT_PBS=$(mktemp)



# Write the script content to the file
cat << OUTER_EOF > $SCRIPT_PBS
#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Run the simulation
$PYTHON <<-INNER_EOF
import luto.simulation as sim
import luto.settings as settings
data = sim.load_data_from_disk('input/Data_RES{settings.RESFACTOR}.pkl')
sim.run(data=data, base=2010, target=2050)
INNER_EOF
OUTER_EOF



# Submit the job to PBS
qsub -N ${JOB_NAME} \
     -q ${QUEUE} \
     -l ncpus=${NCPUS} \
     -l mem=${MEM} \
     -l jobfs=10GB \
     -l walltime=48:00:00 \
     ${SCRIPT_PBS}

# Remove the temporary script file
rm $SCRIPT_PBS