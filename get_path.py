import os

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the directory containing the current script
project_directory = os.path.dirname(current_script_path)

print(project_directory)