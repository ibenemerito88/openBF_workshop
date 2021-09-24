import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import reslast


plt.close("all")

# Symmetric network
q,a,p,u,c,n,s = reslast.resu("network")
# Non-symmetric network
qn,an,pn,un,cn,nn,sn = reslast.resu("networknonsym")





plt.show()