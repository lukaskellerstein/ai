import pprint

def printObject(name, obj):
    print(f"------------------- {name} -------------------")
    print(f"Type: {type(obj)}")
    attributes_dict = vars(obj)
    print("Keys:")
    pprint.pprint(list(attributes_dict.keys()))
    print("\nAttributes:")
    pprint.pprint(attributes_dict)