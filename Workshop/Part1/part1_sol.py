import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import reslast


plt.close("all")

# Part 1: basic running and post-processing
# If added alias you could use os.system("openBF network.yml")


# 1 - Single artery
sing_q, sing_a, sing_p, sing_u, sing_c, names1, t1 = reslast.resu("single")

# Print vessel names


# Compute total flow (in/out)
flowin1 = integrate.simps(sing_q[names1[0]][:,1],t1)
flowout1 = integrate.simps(sing_q[names1[0]][:,5],t1)


# 2 - bifurcation
bif_q, bif_a, bif_p, bif_u, bif_c, bif_names, bif_t = reslast.resu("bifurcation")

# Check conservation of mass at junction
plt.figure()
plt.plot(bif_t,bif_q["1-P"][:,5],label='1-P')
plt.plot(bif_t,bif_q["2-d1"][:,1],label='2-d1')
plt.plot(bif_t,bif_q["3-d2"][:,1],'-.',label='3-d2')
plt.plot(bif_t,bif_q["2-d1"][:,1]+bif_q["3-d2"][:,1],'-.')
plt.legend()
plt.title("Is mass conserved at the junction?")

# Pressure assumption: conservation of static pressure
plt.figure()
plt.plot(bif_t,bif_p["1-P"][:,5],label='1-P')
plt.plot(bif_t,bif_p["2-d1"][:,1],label='2-d1')
plt.plot(bif_t,bif_p["3-d2"][:,1],'-.',label='3-d2')
plt.legend()
plt.title("Is static pressure conserved at the junction?")


# Pressure drop
plt.figure()
plt.plot(bif_t,bif_p["1-P"][:,1]-bif_p["2-d1"][:,5],label='Inlet - outlet 1')
plt.plot(bif_t,bif_p["1-P"][:,1]-bif_p["3-d2"][:,5],'.',label='Inlet - outlet 2')
plt.legend()
plt.title("Pressure drop")



plt.show()