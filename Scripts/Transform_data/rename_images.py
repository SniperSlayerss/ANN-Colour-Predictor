import os
os.getcwd()
collection = "Images"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("Images/" + filename, str(i) + ".jpg")