#!/bin/bash

# Read the settings_bash file ==> MEM, TIME, THREADS, JOB_NAME, HPC_TYPE
source luto/settings_bash.py
export PATH=$PATH:/usr/local/bin
NODE=$(hostname)

if [ -f ~/gurobi.lic ]; then
    grb_license=~/gurobi.lic
else
    grb_license=~/gurobi_${NODE}.lic
fi

export GRB_LICENSE_FILE=$grb_license


# Create a temporary script file
SCRIPT_SLURM=$(mktemp)
SCRIPT_PBS=$(mktemp)


# Write the script content to the file
cat << OUTER_EOF > $SCRIPT_SLURM
#!/bin/bash
# Activate the Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate luto

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Run the simulation
python <<-INNER_EOF
import luto.simulation as sim
data = sim.load_data()
sim.run(data=data, base=2010, target=2050)
from luto.tools.write import write_outputs
write_outputs(data)
INNER_EOF
OUTER_EOF



cat << OUTER_EOF > $SCRIPT_SLURM
#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Activate the Conda environment
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



# Define the script content based on the HPC system type
if [ "$HPC_TYPE" = "SLURM" ]; then
    # Submit the job to SLURM
    sbatch -p ${NODE} \
        --time=${TIME} \
        --mem=${MEM} \
        --cpus-per-task=${CPU_PER_TASK} \
        --job-name=${JOB_NAME} \
        ${SCRIPT_SLURM}
elif [ "$HPC_TYPE" = "PBS" ]; then
    # Submit the job to PBS
    qsub -l nodes=1: \
         ppn=${CPU_PER_TASK} \
         -l walltime=${TIME} \
         -l mem=${MEM} \
         -N ${JOB_NAME} \
         -q ${NODE} \
         ${SCRIPT_SLURM}
else
    echo "Unsupported HPC system type: $HPC_TYPE"
    echo "Please set the HPC_TYPE variable to either 'SLURM' or 'PBS'"
    exit 1
fi

# Remove the temporary script file
rm $SCRIPT_SLURM
rm $SCRIPT_PBS