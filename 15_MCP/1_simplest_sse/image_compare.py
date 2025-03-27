with open("my_server/resources/test_image.png", "rb") as f1, open("my_client/saved_image.png", "rb") as f2:
    a = f1.read()
    b = f2.read()
    print("Same length:", len(a) == len(b))
    print("Same bytes:", a == b)