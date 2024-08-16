import os
from glob import glob

from luto.tools.Manual_jupyter_books.helpers import add_meta_to_nb

print(os.getcwd())



nb_paths = glob('*.ipynb')
print(nb_paths)
for nb_path in nb_paths:
    add_meta_to_nb(nb_path)