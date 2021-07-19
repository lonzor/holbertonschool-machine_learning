#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()

# basic line graph
graph1 = fig.add_subplot(3, 2, 1)
graph1.plot(y0, 'r')
graph1.set_xlim((0, 10))

# scatter point
graph2 = fig.add_subplot(3, 2, 2)
graph2.set_title("Men's Height vs Weight", fontsize='x-small')
graph2.set_xlabel('Height (in)', fontsize='x-small')
graph2.set_ylabel('Weight (lbs)', fontsize='x-small')
graph2.scatter(x1, y1, c='m')

# logarithmically scaled
graph3 = fig.add_subplot(3, 2, 3)
graph3.set_title("Exponential Decay of C-14", fontsize='x-small')
graph3.plot(x2, y2)
graph3.set_xlabel('Time (years)', fontsize='x-small')
graph3.set_ylabel('Fraction Remaining', fontsize='x-small')
graph3.set_yscale("log")
graph3.set_xlim((0, 28650))

# compare 2 elements with a legend
g4 = fig.add_subplot(3, 2, 4)
g4.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
g4.set_xlabel('Time (years)', fontsize='x-small')
g4.set_ylabel('Fraction Remaining', fontsize='x-small')
g4.plot(x3, y31, 'r--', label='C-14')
g4.plot(x3, y32, 'g--', label='Ra-226')
g4.set_xlim((0, 20000))
g4.set_ylim((0, 1))
g4.legend()

# histogram
g5 = fig.add_subplot(3, 1, 3)
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
g5.set_title("Project A", fontsize='x-small')
g5.set_xlabel('Grades', fontsize='x-small')
g5.set_ylabel('Number of Students', fontsize='x-small')
g5.set_ylim(0, 30)
g5.set_xlim(0, 100)
g5.hist(student_grades, bins, edgecolor='black')
g5.set_xticks(np.arange(0, 101, 10))

fig.suptitle("All in One")
plt.tight_layout()
plt.show()
