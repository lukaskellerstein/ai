import os
import argparse

parser = argparse.ArgumentParser(description="My Job")
parser.add_argument("--data", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

# will print a path to the file
print(args.data)

file_content = ""
# will print the contents of the file
with open(args.data + "/file.txt", "r") as f:
    file_content = f.read()
    print(file_content)


# change the text from the file
file_content = file_content.replace("Hello", "Goodbye")

print(file_content)

# create output and write change text to file
with open(os.path.join(args.output_dir, "file.txt"), "w") as f:
    f.write(file_content)
