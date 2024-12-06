import os
import shutil
import luto.simulation as sim
import luto.settings as settings


# Load data 
if os.path.exists(f"{settings.INPUT_DIR}/Data_RES{settings.RESFACTOR}.pkl"):
    data = sim.load_data_from_disk(f"{settings.INPUT_DIR}/Data_RES{settings.RESFACTOR}.pkl")
else:
    data = sim.load_data()

# Run simulation
sim.run(data=data, base=2010, target=2050)


# Move report dir to the project root, only works in Linux system
report_dir = f"{data.path}/DATA_REPORT"
destination_dir ='./DATA_REPORT'
shutil.move(report_dir, destination_dir)

# Remove all files in the project root except the DATA_REPORT dir
for item in os.listdir('.'):
    if item != 'DATA_REPORT':
        try:
            if os.path.isfile(item) or os.path.islink(item):
                os.unlink(item)  # Remove the file or link
            elif os.path.isdir(item):
                shutil.rmtree(item)  # Remove the directory
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")