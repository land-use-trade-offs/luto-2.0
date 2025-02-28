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


TASK_ROOT_DIR = '../Custom_runs/20250228_RES15_Timeseries_DIFF_BIO_PENALTIES'  
if TASK_ROOT_DIR[-1] == '/':
    TASK_ROOT_DIR = TASK_ROOT_DIR[:-1]



EXCLUDE_DIRS = [
    'input', 
    'output', 
    '.git', 
    '.vscode', 
    '__pycache__', 
    'jinzhu_inspect_code'
]

GHG_ORDER = [
    '1.5C (67%) excl. avoided emis', 
    '1.5C (50%) excl. avoided emis', 
    '1.8C (67%) excl. avoided emis'
]

BIO_TARGET_ORDER = [
    '{2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3}', 
    '{2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5}'
]