import numpy as np
import matplotlib.pyplot as plt

labels = []
average_time = []
number_of_iterations = 10

filename = 'exectime.txt'

with open(filename, 'r') as reader:
    i = 0
    avg = 0
    for line in reader:
        if i == 0:
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

print(labels, average_time)
fig, ax = plt.subplots()
ax.barh(x, average_time)
ax.set_title('Average execution times ('+str(number_of_iterations)+' iterations)')
plt.xlabel('Average execution times $[s]$')
ax.set_yticks(x)
ax.set_yticklabels(labels)
plt.show()
