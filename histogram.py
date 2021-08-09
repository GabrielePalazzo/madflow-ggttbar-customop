import numpy as np
import matplotlib.pyplot as plt

labels = []
average_time = []
number_of_iterations = 10

filename = 'bak_32.txt'
#filename = 'bak_256.txt'
#filename2 = 'bak_32.txt'
"""
with open(filename, 'r') as reader:
    f = open(filename2)
    i = 0
    avg = 0
    av2 = 0
    for line in reader:
        l2 = f.readline()
        if i == 0:
            if filename == 'bak_256.txt':
                cuda_string = 'custom op cuda '
                cut_length = len(cuda_string)
                if line[:cut_length] == cuda_string:
                    line = line[cut_length:]
                if l2[:cut_length] == cuda_string:
                    l2 = l2[cut_length:]
            labels.append(str(line))
            labels.append(str(l2))
            i += 1
        elif i == number_of_iterations:
            average_time.append(avg/number_of_iterations)
            average_time.append(av2/number_of_iterations)
            i = 0
            avg = 0
            av2 = 0
        else:
            avg += float(line)
            av2 += float(l2)
            i += 1
"""

with open(filename, 'r') as reader:
    i = 0
    avg = 0
    for line in reader:
        if i == 0:
            cuda_string = 'custom op cuda '
            cut_length = len(cuda_string)
            if line[:cut_length] == cuda_string:
                line = line[cut_length:]
            labels.append(str(line))
            i += 1
        elif i == number_of_iterations:
            average_time.append(avg/number_of_iterations)
            i = 0
            avg = 0
        else:
            avg += float(line)
            i += 1

print('Done')

N = len(labels)
x = np.arange(N)

#print(labels, average_time)
fig, ax = plt.subplots()
ax.barh(x, average_time)
#ax.set_title('Average execution times ('+str(number_of_iterations)+' iterations) Blue: 256 threads per block '+'Red: 32 threads per block')
ax.set_title('Average execution times ('+str(number_of_iterations)+' iterations) 32 threads per block')
#ax.set_title('Average execution times ('+str(number_of_iterations)+' iterations) custom op: cuda - 32 threads per block / openmp - 12 threads')
plt.xlabel('Average execution times $[s]$')
"""
for i in range(N):
    if i%2 == 1:
        ax.get_children()[i].set_color('r')
    i += 1
"""    

ax.set_yticks(x)
ax.set_yticklabels(labels)
plt.show()
#plt.savefig("hist.pdf",bbox_inches='tight',pad_inches=0)
