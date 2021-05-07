import matplotlib.pyplot as plt
import numpy as np
import util

Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
print(Xa[:, 1:])
plt.figure()
util.plot_points(Xa[:, 1:], Ya)
plt.savefig('ds1_a.png')
Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
plt.figure()
util.plot_points(Xb[:, 1:], Yb)
plt.savefig('ds1_b.png')
