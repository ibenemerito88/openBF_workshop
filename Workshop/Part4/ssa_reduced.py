import numpy as np
import os
import GPy
import sys
from SALib.sample import saltelli
from SALib.analyze import sobol

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
#from parfor import parfor
import time
from scipy import stats

### SENSITIVITY ANALYSIS
"""
40% everywhere
"""

inpnames = ['3-D_R0', '3-D_E', '4-F_R0', '4-F_E', '5-F_R0', '5-F_E', '6-F_R0', '6-F_E', '6-F_R1', '6-F_Cc', '7-F_R0', '7-F_E',
 '7-F_R1', '7-F_Cc', '8-F_R0', '8-F_E', '8-F_R1', '8-F_Cc', '9-F_R0', '9-F_E', '9-F_R1', '9-F_Cc']
outnames = ['Qmin-1','Qmin-2','Qmin-3','Qmean-1','Qmean-2','Qmean-3','Qmax-1','Qmax-2','Qmax-3',
'Pmin-1','Pmin-2','Pmin-3','Pmean-1','Pmean-2','Pmean-3','Pmax-1','Pmax-2','Pmax-3',
'Umin-1','Umin-2','Umin-3','Umean-1','Umean-2','Umean-3','Umax-1','Umax-2','Umax-3']



def computemape(A,F):
	mape = np.zeros([np.shape(A)[1]])
	for j in range(np.shape(A)[1]):
		for i in range(np.shape(A)[0]):
			mape[j] += np.abs((A[i,j]-F[i,j])/A[i,j])
	return mape/(np.shape(A)[0])



plt.close("all")
init=os.getcwd()

print("### TRAIN THE EMULATOR ###")
print("--- Import and normalise inputs ---")
# Import inputs and outputs
IINP=np.loadtxt("TRAIN_IINP.txt")
# Find input vessels
vves = []
for i in range(IINP.shape[1]):
	if np.std(IINP[:,i])>1e-12:
		vves.append(i)
vves = np.array(vves)
IINP = IINP[:,vves]
normIINP=np.zeros([1,IINP.shape[1]])
for i in range(IINP.shape[1]):
	normIINP[0,i]=np.linalg.norm(IINP[:,i])
INP=IINP/normIINP
print("--- Import and normalise outputs ---")
OOUT=np.loadtxt("TRAIN_OOUT.txt")
normOOUT=np.zeros([1,OOUT.shape[1]])
for i in range(OOUT.shape[1]):
	normOOUT[0,i]=np.linalg.norm(OOUT[:,i])
OUT=OOUT/normOOUT


############## REDUCED SENSITIVITY
if not os.path.isdir("sensitivity_reduced"):
	os.mkdir("sensitivity_reduced")
os.chdir("sensitivity_reduced")

inpnames = ['3-D_R0', '3-D_E', '4-F_R0', '4-F_E', '6-F_R0', '6-F_E', '6-F_R1', '6-F_Cc']

red = [0,1,2,3,6,7,8,9]

IINP = IINP[:,red]
normIINP=np.zeros([1,IINP.shape[1]])
for i in range(IINP.shape[1]):
	normIINP[0,i]=np.linalg.norm(IINP[:,i])
INP=IINP/normIINP

nvar = INP.shape[1]	# number of input variables
ker1 = GPy.kern.RBF(nvar) #+GPy.kern.Matern52(nvar, ARD=True)
m1 = GPy.models.GPRegression(INP,OUT,ker1)
m1.optimize()
m1.optimize_restarts(num_restarts = 5)

problem={'num_vars':INP.shape[1],'bounds':np.column_stack([np.min(INP,axis=0),np.max(INP,axis=0)])}
param_values = saltelli.sample(problem,1000,calc_second_order=True)
y,v=m1.predict(param_values)

Sis=[]
### RUN SENSITIVITY
tic=time.perf_counter()
for i in range(y.shape[1]):
	Si=sobol.analyze(problem,y[:,i],calc_second_order=True)
	print("Index: "+str(i+1)+"/"+str(y.shape[1]))
	Sis.append(Si)
	np.savetxt("Si1-"+str(i+1)+".txt",Si['S1'])
	np.savetxt("Si2-"+str(i+1)+".txt",Si['S2'])
	np.savetxt("SiT-"+str(i+1)+".txt",Si['ST'])
toc=time.perf_counter()
print("Elapsed time: "+str(toc-tic))

s1=[]
st=[]
for i in range(y.shape[1]):
    s1.append(np.loadtxt("Si1-"+str(i+1)+".txt"))
    st.append(np.loadtxt("SiT-"+str(i+1)+".txt"))

s1=np.array(s1)
st=np.array(st)
sns.set(style="dark")
os.chdir(init)

plt.figure()
cmap=sns.cubehelix_palette(8,start=.5,rot=.75,hue=0,as_cmap=1)
sns.heatmap(st,cmap=cmap,xticklabels=inpnames,yticklabels=outnames)
plt.xticks(rotation='vertical')
plt.title("Sensitivity of distal perfusion to input parameters")
#plt.close("all")


### study behaviour
for i in range(1,INP.shape[1]):
	INP[:,i]=np.mean(INP[:,i])
Y,V = m1.predict(INP)
Y=Y*normOOUT


plt.show()
