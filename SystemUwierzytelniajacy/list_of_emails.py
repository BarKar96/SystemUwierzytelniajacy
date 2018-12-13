f = open ("emails.txt", "r")
f1 = f.readlines()
global listOfTuples
listOfTuples= []
for x in f1:
    list = x.split()
    listOfTuples.append(list)