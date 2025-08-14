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

import os
import shutil
import zipfile
import luto.simulation as sim
import luto.settings as settings



# Run the simulation
data = sim.load_data()
sim.run(data=data)


# Set up report directory and archive path
report_dir = f"{data.path}/DATA_REPORT"
archive_path ='./DATA_REPORT.zip'

data = None  # Clear the data object to free memory
sim = None   # Clear the simulation module to free memory

# Zip the output directory, and remove the original directory
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(report_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, start=report_dir)
            zipf.write(abs_path, arcname=rel_path)

# Remove all files except the report directory
for item in os.listdir('.'):
    if item != 'DATA_REPORT.zip':
        try:
            if os.path.isfile(item) or os.path.islink(item):
                os.unlink(item)  # Remove the file or link
            elif os.path.isdir(item):
                shutil.rmtree(item)  # Remove the directory
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")
            
