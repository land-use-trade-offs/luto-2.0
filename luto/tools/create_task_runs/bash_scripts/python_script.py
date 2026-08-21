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

# Usage:
#   python python_script.py                          # use INPUT_DIR already in luto/settings.py
#   python python_script.py --input_dir /path/to/input  # override INPUT_DIR before running

import argparse, re, pathlib, os, sys
import shutil
import zipfile

# Force UTF-8 on Windows consoles (default cp1252 can't handle box-drawing chars).
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def patch_input_dir(input_dir: str):
    settings_path = pathlib.Path('luto/settings.py')
    new_dir = pathlib.Path(input_dir).as_posix()
    text = settings_path.read_text(encoding='utf-8')

    # Replace INPUT_DIR=... line only — NO_GO_VECTORS uses relative paths so needs no patching
    text = re.sub(r'^INPUT_DIR\s*=.*$', f'INPUT_DIR="{new_dir}"', text, flags=re.MULTILINE)

    settings_path.write_text(text, encoding='utf-8')


# MUST parse args and patch settings BEFORE importing luto modules,
# because parameters.py reads INPUT_DIR at module load time.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None, help='Override INPUT_DIR in luto/settings.py')
    args = parser.parse_args()
    if args.input_dir:
        patch_input_dir(args.input_dir)

import luto.simulation as sim
import luto.settings as settings

# If a checkpoint exists, restore the original timestamp BEFORE load_data() so
# sim.run() reuses the existing output directory. load_data() is skipped —
# sim.run() loads the checkpoint internally when data=None.
_checkpoint_dir = next(
    (str(d) for d in sorted(pathlib.Path(settings.OUTPUT_DIR).iterdir(), key=lambda d: d.name)
     if d.is_dir() and any(re.match(r'data_\d{4}\.lz4', f.name) for f in d.iterdir())),
    None
)
data = None if _checkpoint_dir else sim.load_data()


# Get data and run the simulation, catching any errors to ensure the run dir 
# is archived and cleaned up before exiting non-zero.
try:
    data = sim.run(data=data, do_report=settings.WRITE_OUTPUTS, checkpoint_dir=_checkpoint_dir)
    sim_error = None
    simulation_root = pathlib.Path(data.path).absolute().parent.parent
except Exception as e:
    sim_error = e
    # data.path can't be relied upon - the run dir is simply the parent of OUTPUT_DIR.
    simulation_root = pathlib.Path(settings.OUTPUT_DIR).absolute().parent


# Set up report directory and archive path
run_idx = simulation_root.name
report_data_dir = simulation_root.parent / 'Report_Data'
report_data_dir.mkdir(parents=True, exist_ok=True)

report_zip_path = report_data_dir / f'{run_idx}.zip'
archive_path = simulation_root / 'Run_Archive.zip'


# Walk the run dir once, splitting files into the main archive and the
# DATA_REPORT subtree (which only exists if the simulation succeeded).
files_run, files_report = [], []
for root, dirs, files in os.walk(simulation_root):
    for file in files:
        if file == 'Run_Archive.zip':
            continue
        abs_path = pathlib.Path(root) / file
        if 'DATA_REPORT' in abs_path.as_posix():
            files_report.append(abs_path)
        else:
            files_run.append(abs_path)

with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as run_zip:
    for abs_path in files_run:
        run_zip.write(abs_path, arcname=abs_path.relative_to(simulation_root))

if files_report:
    with zipfile.ZipFile(report_zip_path, 'w', zipfile.ZIP_DEFLATED) as report_zip:
        for abs_path in files_report:
            # arcname is the path starting from (and including) DATA_REPORT
            parts = abs_path.parts
            arcname = pathlib.Path(*parts[parts.index('DATA_REPORT'):])
            report_zip.write(abs_path, arcname=arcname)


# Remove all files after archiving
for item in os.listdir(simulation_root):
    if item != 'Run_Archive.zip':
        item_path = simulation_root / item
        try:
            if item_path.is_file() or item_path.is_symlink():
                item_path.unlink()
            elif item_path.is_dir():
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


# Re-raise the simulation error after archiving so the job exits non-zero.
if sim_error is not None:
    raise sim_error

