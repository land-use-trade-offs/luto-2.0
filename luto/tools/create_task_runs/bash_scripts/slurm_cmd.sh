#!/bin/bash

# Read the settings_bash file ==> MEM, TIME, THREADS, JOB_NAME
source luto/settings_bash.py


# Create a temporary script file
SCRIPT=$(mktemp)

# Write the script content to the file
cat << OUTER_EOF > $SCRIPT
#!/bin/bash
# Activate the Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
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
    --cpus-per-task=${THREADS} \
    --export=ALL \
    --job-name=${JOB_NAME} \
    ${SCRIPT}

# Remove the temporary script file
rm $SCRIPT