# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

import os, pathlib
import shutil
import zipfile
import luto.simulation as sim



# Run the simulation
data = sim.load_data()
sim.run(data=data)


# Set up report directory and archive path
output_dir = pathlib.Path(data.path).absolute()
simulation_root = output_dir.parent.parent

run_idx = simulation_root.name
report_data_dir = simulation_root.parent / 'Report_Data'
report_data_dir.mkdir(parents=True, exist_ok=True)

report_zip_path = report_data_dir / f'{run_idx}.zip'
archive_path = output_dir.parent.parent / 'Run_Archive.zip'



# Zip the output directory, and remove the original directory
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as run_zip,\
    zipfile.ZipFile(report_zip_path, 'w', zipfile.ZIP_DEFLATED) as report_zip: 
    for root, dirs, files in os.walk(simulation_root): 
        files = [f for f in files if f != 'Run_Archive.zip']  # Exclude existing zip files
        for file in files:
            abs_path = pathlib.Path(root) / file 
            if 'DATA_REPORT' in abs_path.as_posix():
                zip_path = abs_path.relative_to(output_dir)
                report_zip.write(abs_path, arcname=zip_path)
            else:
                zip_path = abs_path.relative_to(simulation_root)
                run_zip.write(abs_path, arcname=zip_path)


# Remove all files after archiving
for item in os.listdir(simulation_root):
    if item != 'Run_Archive.zip':
        try:
            if os.path.isfile(item) or os.path.islink(item):
                os.unlink(item)  
            elif os.path.isdir(item):
                shutil.rmtree(item) 
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")
            
