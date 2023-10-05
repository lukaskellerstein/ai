# filename: read_project_info.py

import os

project_dir = 'temp_dir_0/fluentui-app'
readme_file = os.path.join(project_dir, 'README.md')

if os.path.exists(readme_file):
    with open(readme_file, 'r') as file:
        print(file.read())
else:
    print(f"No README file found in {project_dir}")