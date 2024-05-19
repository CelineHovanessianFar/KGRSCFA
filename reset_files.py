

import shutil
import os

directories_to_delete = ['CAFE/tmp', 'tmp']

for directory in directories_to_delete:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Deleted {directory}")
    else:
        print(f"Directory {directory} does not exist")
