import os

in_dest = "/media/MyDrive/Project Fake news/Data/new data/True/Sejuti-unformat/রাজনীতি"
out_dest = "/media/MyDrive/Project Fake news/Data/new data/True/Himel/অপরাধ ও আইন"


inFiles = os.listdir(in_dest)
outFiles = os.listdir(out_dest)
i = 0
for f in outFiles:
    number = f.split(" ")[1]
    newName = "অপরাধ ও আইন_"+number
    os.rename(out_dest + "/" + f, out_dest+"/" + newName)
    print(newName)
