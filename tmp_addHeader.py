import os

# Define the copyright header
COPYRIGHT_HEADER = """# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
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
# LUTO2. If not, see <https://www.gnu.org/licenses/>.\n\n"""

# Function to add copyright header to a Python file
def add_header_to_file(file_path):
    with open(file_path, "r+", encoding="utf-8") as f:
        content = f.read()
        if COPYRIGHT_HEADER.strip() in content:  # Check if header is already present
            print(f"Skipping {file_path} (header already exists)")
            return
        f.seek(0, 0)  # Move to the beginning of the file
        f.write(COPYRIGHT_HEADER + content)  # Write header followed by original content
    print(f"Updated {file_path}")

# Function to iterate over all Python files in the project
def process_project_files(root_dir):
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(foldername, filename)
                add_header_to_file(file_path)

# Run the script
if __name__ == "__main__":
    project_root = os.getcwd()  # Set this to your project root if running from another location
    process_project_files(project_root)
    print("Finished adding copyright headers.")