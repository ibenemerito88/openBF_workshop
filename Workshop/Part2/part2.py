import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import reslast


plt.close("all")

# Part 1: basic running and post-processing
# If added alias you could use os.system("openBF network.yml")


# 1 - Single artery E1/E2
q1, a1, p1, u1, names1, t1 = reslast.resu("single_E1")
q2, a2, p2, u2, names2, t2 = reslast.resu("single_E2")


q111, a111, p111, u111, names111, t111 = reslast.resu("bifurcation_R111")
q112, a112, p112, u112, names112, t112 = reslast.resu("bifurcation_R112")

plt.figure()
plt.plot(q111["2-d1"][:,3],label='r=r0')
plt.plot(q112["2-d1"][:,3],label='r=r0')
plt.legend()
plt.figure()
plt.plot(q111["3-d2"][:,3],label='r=r0')
plt.plot(q112["3-d2"][:,3],label='r=r1>r0')
plt.legend()

plt.show()



