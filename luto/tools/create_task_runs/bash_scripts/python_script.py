import os
import shutil
import luto.simulation as sim
import luto.settings as settings


# Run the simulation
data = sim.load_data()
sim.run(data=data, base=2010, target=2050)
sim.save_data_to_disk(data, f"{data.path}/DATA_REPORT/Data_{settings.MODE}_RES{settings.RESFACTOR}.gz")


# Remove all files except the report directory if settings.KEEP_OUTPUTS is False
'''
KEEP_OUTPUTS is not originally defined in the settings, but will be added in the `luto/tools/create_task_runs/create_running_tasks.py` file.
'''

if settings.KEEP_OUTPUTS is False:

    report_dir = f"{data.path}/DATA_REPORT"
    destination_dir ='./DATA_REPORT'
    shutil.move(report_dir, destination_dir)

    for item in os.listdir('.'):
        if item != 'DATA_REPORT':
            try:
                if os.path.isfile(item) or os.path.islink(item):
                    os.unlink(item)  # Remove the file or link
                elif os.path.isdir(item):
                    shutil.rmtree(item)  # Remove the directory
            except Exception as e:
                print(f"Failed to delete {item}. Reason: {e}")