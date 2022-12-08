from matplotlib import pyplot as plt

arr=[]
with open('./result.txt') as f:
    [arr.append(int(line[:-1])) for line in f.readlines()]

print(len(arr))
bins = list(range(1,len(arr)+1))
plt.bar(bins, arr, width=1.0)
plt.show()