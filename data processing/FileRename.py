import os

in_dest = "../Data/Labeled Dataset/Fake News"
out_dest = "../Data/Labeled Dataset/Fake News"


inFiles = os.listdir(in_dest)
outFiles = os.listdir(out_dest)
i = 1
for f in outFiles:
    number = i
    i = i+1
    newName = "Fake-"+str(number)
    os.rename(out_dest + "/" + f, out_dest+"/" + newName)
    # print(newName)
