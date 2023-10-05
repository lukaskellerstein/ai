# filename: list_project_files.py

import os

project_dir = 'temp_dir_0/fluentui-app'

for root, dirs, files in os.walk(project_dir):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))