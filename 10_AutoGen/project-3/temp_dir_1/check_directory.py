# filename: check_directory.py

import os

project_dir = './temp_dir_0/fluentui-app'

if os.path.exists(project_dir):
    if os.listdir(project_dir):
        print(f"The directory {project_dir} exists and is not empty.")
    else:
        print(f"The directory {project_dir} exists but is empty.")
else:
    print(f"The directory {project_dir} does not exist.")