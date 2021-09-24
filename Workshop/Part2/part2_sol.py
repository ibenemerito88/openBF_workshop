import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import reslast


plt.close("all")

### Part 2: investigate the effect of parameters on simple networks


# Radius: single vessel
qr1, ar1, pr1, ur1, cr1, namesr1, tr1 = reslast.resu("single_R1")
qr2, ar2, pr2, ur2, cr2, namesr2, tr2 = reslast.resu("single_R2")

plt.figure()
plt.plot(ar1["1-A1"][:,3],label='r1')
plt.plot(ar2["1-A1"][:,3],label='r2>r1')
plt.legend()
plt.title("Radius: single vessel - Effect on radius")

plt.figure()
plt.plot(pr1["1-A1"][:,3],label='r1')
plt.plot(pr2["1-A1"][:,3],label='r2>r1')
plt.legend()
plt.title("Radius: single vessel - Effect on pressure")

plt.figure()
plt.plot(ur1["1-A1"][:,3],label='r1')
plt.plot(ur2["1-A1"][:,3],label='r2>r1')
plt.legend()
plt.title("Radius: single vessel - Effect on velocity")



# Radius: bifurcation

q111, a111, p111, u111, c111, names111, t111 = reslast.resu("bifurcation_R111")
q112, a112, p112, u112, c112, names112, t112 = reslast.resu("bifurcation_R112")

plt.figure()
plt.plot(q111["1-P"][:,3],label="r=r0")
plt.plot(q112["1-P"][:,3],label="r=r0")
plt.legend()
plt.title("Radius: bifurcation - Flow in 1-P")

plt.figure()
plt.plot(q111["2-d1"][:,3],label='r=r0')
plt.plot(q112["2-d1"][:,3],label='r=r0')
plt.legend()
plt.title("Radius: bifurcation - Flow in 2-d1")

plt.figure()
plt.plot(q111["3-d2"][:,3],label='r=r0')
plt.plot(q112["3-d2"][:,3],label='r=r1>r0')
plt.title("Radius: bifurcation - Flow in 3-d2")
plt.legend()





# Young's modulus: single vessel
qe1, ae1, pe1, ue1, ce1, names1, t1 = reslast.resu("single_E1")
qe2, ae2, pe2, ue2, ce2, names2, t2 = reslast.resu("single_E2")

plt.figure()
plt.plot(ae1["1-A1"][:,3],label='E1')
plt.plot(ae2["1-A1"][:,3],label='E2>E1')
plt.legend()
plt.title("Young's modulus: single vessel - Effect on radius")

plt.figure()
plt.plot(ue1["1-A1"][:,3],label='E1')
plt.plot(ue2["1-A1"][:,3],label='E2>E1')
plt.legend()
plt.title("Young's modulus - Effect on velocity")

plt.figure()
plt.plot(ce1["1-A1"][:,3],label='E1')
plt.plot(ce2["1-A1"][:,3],label='E2>E1')
plt.legend()
plt.title("Young's modulus - Effect on wave speed")






plt.show()



