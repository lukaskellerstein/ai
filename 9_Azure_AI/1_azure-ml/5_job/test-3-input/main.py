import argparse

parser = argparse.ArgumentParser(description="My Job")
parser.add_argument("--data", type=str)
args = parser.parse_args()

# will print a path to the file
print(args.data)

# will print the contents of the file
with open(args.data + "/file.txt", "r") as f:
    print(f.read())
