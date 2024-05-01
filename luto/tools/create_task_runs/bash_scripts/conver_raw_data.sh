# Run the simulation
python <<-INNER_EOF
# Activate the Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate luto
from luto.dataprep import create_new_dataset
create_new_dataset()
INNER_EOF