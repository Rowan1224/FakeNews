import os
Category = input()
dataDirectory = "../Data/Raw/"+Category+"/"
filePaths = os.listdir(dataDirectory)
outputDirectory = "../Data/Dataset/"+Category+"/"
num = 1
Headlines = {"headline"}
for filepath in filePaths:
    data = []
    filepath = dataDirectory + filepath
    output = outputDirectory+Category+"-" + str(format(num, '04d'))+".txt"
    num = num + 1
    with open(filepath, 'r', encoding='utf8') as infile, open(output, "w", encoding='utf8') as outfile:
        for line in infile:
            data.append(line)
        if len(data) < 8:
            print(filepath)
            continue

        if data[6] in Headlines:
            continue
        else:
            Headlines.add(data[0])
            article = ""
            for i in range(7, len(data)):
                body = data[i].replace("\n","")
                article = article + body
            data[7] = article
            for i in range(8):
                outfile.write(data[i])