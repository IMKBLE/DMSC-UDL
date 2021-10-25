a = []
for i in range(300):
    a.append(i)

b = []
for i in range(a.__len__()):
    if((i+1)%6 != 0):
        b.append(a[i])
print(b)