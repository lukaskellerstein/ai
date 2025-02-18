import argparse

parser = argparse.ArgumentParser(description="My Job")
parser.add_argument("--data", type=str)
args = parser.parse_args()

print(args.data)
