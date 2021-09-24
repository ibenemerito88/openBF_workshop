import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import reslast


plt.close("all")

# Part 1: basic running and post-processing
# If added alias you could use os.system("openBF network.yml")


# 1 - Single artery
sing_q1, sing_a1, sing_p1, sing_u1, names1, t1 = reslast.resu("single")

# Print vessel names


# Compute total flow (in/out)
flowin1 = integrate.simps(sing_q1[names1[0]][:,1],t1)
flowout1 = integrate.simps(sing_q1[names1[0]][:,5],t1)










# 2 - bifurcation
bif_q, bif_a, bif_p, bif_u, bif_names, bif_t = reslast.resu("bifurcation")
plt.figure()
plt.plot(bif_t,bif_q["1-P"][:,5],label='1-P')
plt.plot(bif_t,bif_q["2-d1"][:,1],label='2-d1')
plt.plot(bif_t,bif_q["3-d2"][:,1],'-.',label='3-d2')
plt.plot(bif_t,bif_q["2-d1"][:,1]+bif_q["3-d2"][:,1],'-.')
plt.legend()






# 3 - conjunction
conj_q, conj_a, conj_p, conj_u, conj_names, conj_t = reslast.resu("conjunction")

plt.show()