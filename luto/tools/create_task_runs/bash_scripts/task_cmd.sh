#!/bin/bash

# Read the settings_bash file ==> JOBNAME, QUEUE, NCPUS, MEM, NCPUS, QUEUE
source luto/settings_bash.py

# Create a temporary script file
SCRIPT_PBS=$(mktemp)


# Write the script content to the file
cat << OUTER_EOF > $SCRIPT_PBS
#!/bin/bash

#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l ncpus=${NCPUS}
#PBS -l mem=${MEM}
#PBS -l jobfs=10GB
#PBS -l walltime=48:00:00

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Activate the Conda environment
conda activate luto

# Run the simulation
python <<-INNER_EOF
import luto.simulation as sim
import luto.settings as settings
data = sim.load_data_from_disk('input/Data_RES{settings.RESFACTOR}.pkl')
sim.run(data=data, base=2010, target=2050)
INNER_EOF
OUTER_EOF



# Submit the job to PBS
qsub $SCRIPT_PBS

# Remove the temporary script file
rm $SCRIPT_PBS