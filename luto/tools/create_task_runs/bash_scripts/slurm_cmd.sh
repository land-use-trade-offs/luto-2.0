#!/bin/bash

# Read the settings_bash file
source luto/settings_bash.py


# Set the memory and time based on the resolution factor
if [ "$RESFACTOR" -eq 1 ]; then
    MEM="200G"
    TIME="30-0:00:00"
elif [ "$RESFACTOR" -eq 2 ]; then
    MEM="150G"
    TIME="10-0:00:00"
elif [ "$RESFACTOR" -eq 3 ]; then
    MEM="100G"
    TIME="5-0:00:00"
else
    MEM="80G"
    TIME="2-0:00:00"
fi



# Get the job name based on the task directory
SCRIPT_PATH=$(realpath $(dirname "$0"))
PARENT_DIR=$(basename "${SCRIPT_PATH}")
JOB_NAME="LUTO_${PARENT_DIR}"

# Get the number of CPUs
CPU_COUNT=$(nproc)

# Calculate CPU_PER_JOB
CPU_PER_JOB=$(($CPU_COUNT / $DIR_COUNT - 2))

# Ensure CPU_PER_JOB is not less than 1
if (( CPU_PER_JOB < 1 )); then
    CPU_PER_JOB=1
fi




# Create a temporary script file
SCRIPT=$(mktemp)

# Write the script content to the file
cat << OUTER_EOF > $SCRIPT
#!/bin/bash
# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
source ~/miniforge3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate luto

# Run the simulation
python <<-INNER_EOF
import luto.simulation as sim
data = sim.load_data()
sim.run(data=data, base=2010, target=2050)
from luto.tools.write import write_outputs
write_outputs(data)
INNER_EOF
OUTER_EOF

# Submit the job
sbatch -p mem \
    --time=${TIME} \
    --mem=${MEM} \
    --cpus-per-task=${CPU_PER_JOB} \
    --job-name=${JOB_NAME} \
    ${SCRIPT}

# Remove the temporary script file
rm $SCRIPT