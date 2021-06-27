
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from numpy import sign
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import math
import random


# In[2]:


def ML_vf(x,Iapp):
    # Morris-Lecar model model of an excitable barnacle muscle fiber
    # adapted from Morris and Lecar (1981) Biophysical Journal 35 pp. 193-231
    #      Input:
    #                        x = the position variables and Isyn.
    #                        x = (v,w) where v is the membrane potential and
    #                        w is a gating variable for Ca.
    #                       Isyn isn the synaptic input
    #       Output:
    #                       dx = the x derivative.
    #Iapp=42;

    # 2. parameters for the MorrisLecar-02: at Iapp=Iapp*:=39.96, it undergoes a SNIC bifurcation (a limit cycle appears for Iapp>Iapp*).
    #Info per a escalar input: Amplitud òrbita periòdica per a Iapp=40 (prop de SNIC a Iapp=39.96): 88mV aprox
    #HB, subcritical: 97.79
    #Bistability region in [97.79,116.1], no biologically plausible
    
    Cm=20.0
    phi=0.066667
    
    gCa=4.0
    gL=2.0
    gK=8.0
    
    vCa=120.0
    vL=-60.0
    vK=-84.0
    
    v1=-1.2
    v2=18
    v3=12
    v4=17.4
    
    v= x[0]
    w= x[1]
    Isyn= x[2]
    
    # leak current 
    iL=gL*(v-vL)
    
    # calcium current
    tan1= np.tanh((v-v1)/v2)
    tan2= np.tanh((v-v3)/v4)
    cos2= np.cosh((v-v3)/(2*v4))
    minf=0.5*(1+tan1)
    winf=0.5*(1+tan2)
    iCa=gCa*minf*(v-vCa)
    iK=gK*w*(v-vK)
    
    # neuron dynamics
    vp=(-iL-iCa-iK+Iapp+Isyn)/Cm;
    wp=phi*(winf-w)*cos2;
    
    return [vp,wp]
           


# In[3]:


def f_g(x,Iapp):
    
    Cm=20.0
    phi=0.066667
    
    gCa=4.0
    gL=2.0
    gK=8.0
    
    vCa=120.0
    vL=-60.0
    vK=-84.0
    
    v1=-1.2
    v2=18
    v3=12
    v4=17.4
    
    v= x[0]
    w= x[1]
    
    iL=gL*(v-vL)
    tan1= np.tanh((v-v1)/v2)
    tan2= np.tanh((v-v3)/v4)
    cos2= np.cosh((v-v3)/(2*v4))
    minf=0.5*(1+tan1)
    winf=0.5*(1+tan2)
    iCa=gCa*minf*(v-vCa)
    iK=gK*w*(v-vK)
    
    vp=-iL-iCa-iK+Iapp
    wp=phi*(winf-w)*cos2
    
    return [vp,wp]
    
    


# In[4]:


def fsmd_vf_22_w(x,f): #Filter function (2x2)
    
    v1,v2,v3,v4 = -1.2, 18, 12, 17.4    
    phi=0.066667
    ls0,ls1,ls2,ls3,ls4 = 1.1,4.57,9.30,10.03,5
    L = 2 #Lipschitz constant for dotv
    x1,x2 = x[0],x[1]
    y0,y1,y2 = x[2],x[3],x[4]
    w = x[5]
    
    #Dynamics
    x1p=-ls4*(L**(1/5))*(abs(x1)**(4/5))*sign(x1)+x2
    x2p=-ls3*(L**(2/5))*(abs(x1)**(3/5))*sign(x1)+y0-f
    y0p=-ls2*(L**(3/5))*(abs(x1)**(2/5))*sign(x1)+y1
    y1p=-ls1*(L**(4/5))*(abs(x1)**(1/5))*sign(x1)+y2
    y2p=-ls0*L*sign(x[1])
    
    #wp auxiliary functions
    tan1= np.tanh((y0-v1)/v2)
    tan2= np.tanh((y0-v3)/v4)
    cos2= np.cosh((y0-v3)/(2*v4))
    minf=0.5*(1+tan1)
    winf=0.5*(1+tan2)
    
    #w dynamics
    wp = phi*(winf-w)*cos2
    
    dx = [x1p,x2p,y0p,y1p,y2p,wp]#y2p is vdoubledot
    
    return dx
    
    
    


# In[5]:


'''
Simulation of the Morris-Lecar system using a set of gEs and gIs prepared a priori
'''


# In[6]:


global Iapp, vE, vI, h, tmax

Iapp= 50
vE=10 #0
vI=-50 #-80
h = 0.01    
tmax = 1000
N=int(tmax/h)

with open("sd93_01_gE_gI_0p01.dat") as f:
    lines = [[float(x) for x in line.split()] for line in f]
    
with open("diagramML2.dat") as d:
    dlines = [[float(x) for x in line.split()] for line in d ]

dlen = len(dlines)

gE = [0 for i in range(N)]
gI = [0 for i in range(N)]
Iapp0 = [0 for i in range(dlen)]
maxV = [0 for i in range(dlen)]
minV = [0 for i in range(dlen)]

for i in range(N):
    gE[i] = lines[i][1]
    gI[i] = lines[i][2]
    
for i in range(dlen):
    Iapp0[i] = dlines[i][0]
    maxV[i] = dlines[i][1]
    minV[i] = dlines[i][2]
    
v = [0 for i in range(N)]
w = [0 for i in range(N)]
vdot = [0 for i in range(N)]
Isyn = [0 for i in range(N)]

v[0] = -50
w[0] = 0.1

t = 0 

for i in range(1,N):
    Isyn[i-1]=gE[i-1]*(vE-v[i-1])+gI[i-1]*(vI-v[i-1])
    x0=[v[i-1],w[i-1],Isyn[i-1]]
    #vector field evaluation
    dx=ML_vf(x0,Iapp)
    #Euler integrator (we could improve it using a RK2 method)
    v[i] = v[i-1] + h*dx[0]
    w[i] = w[i-1] + h*dx[1]
    vdot[i] = dx[0]
    #time updates
    t=t+h


It = [Iapp-x for x in Isyn]


# In[7]:


'''
Voltage and Synaptic-Input(Computed from the set of gEs and gIs used)
'''


# In[8]:


#figures

tvec = np.arange(0,tmax,h)

plt.figure(1)
plt.plot(tvec,v,"r")
plt.title('Voltage trace')
plt.xlabel('time (ms)') 
plt.ylabel('voltage (mV)')

plt.figure(2)
plt.plot(tvec,Isyn,'r');
plt.title('Isyn')
plt.xlabel('time (ms)') 
plt.ylabel('current (\mu A/cm^2)')


# In[9]:


'''
Plotting the voltage(one point per each time unit) on the Bifurcation diagram, along with the gEs and gIs
'''


# In[10]:


plt.figure(3)
plt.plot(Iapp0,maxV,label="maxV")
plt.plot(Iapp0,minV,label="minV")
plt.plot(It,v,'b')
plt.xlabel('Iapp (\mu A/cm^2)') 
plt.ylabel('voltage (mV)')


plt.figure(4);
plt.plot(tvec,gE,'b',label="gE");
plt.plot(tvec,gI,'r',label="gI");
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('conductance (mS/cm^2)')


# In[11]:


'''
Analysis of the voltage on the bifurcation diagram using a set of gEs and gIs sampled from the previous set,
such that there are fixed values for intervals of a given size, with the total simulation time unchanged.

Note that subsequent sections of the notebook do not use V and W values from this simulation.
'''


# In[20]:


#Simulation of ML system using constant values of gE and gI for a certain interval length

gEmax, gEmin = max(gE) , min(gE)
gImax, gImin = max(gI) , min(gI)


gEs = [0 for i in range(N)]
gIs = [0 for i in range(N)]

# In order to simulate slower variation, 
# instead of having different gI and gE values for EACH timestep,
# we simulate constant gE and gI values for subintervals of N(Total time)
# we can do this for an interval of size k(every k units of time have the same gE and gI)

# simulate for k=10,20,40
k = 2000
Is = int(N/k)
print("%d time units(ms)" % N)
print("%d intervals" % Is)


for i in range(0,N,k):
    gEs[i] = random.uniform(gEmin,gEmax)
    gIs[i] = random.uniform(gImin,gImax)
    for j in range(1,k):
        gEs[i+j] = gEs[i]
        gIs[i+j] = gIs[i]
        

t = 0 
v = [0 for i in range(N)]
w = [0 for i in range(N)]
Isyn = [0 for i in range(N)]
v[0] = -50
w[0] = 0.05

for i in range(1,N):
    Isyn[i-1]=gEs[i-1]*(vE-v[i-1])+gIs[i-1]*(vI-v[i-1])
    x0=[v[i-1],w[i-1],Isyn[i-1]]
    dx=ML_vf(x0,Iapp)
    v[i] = v[i-1] + h*dx[0]
    w[i] = w[i-1] + h*dx[1]
    t=t+h

It = [Iapp-x for x in Isyn]


# In[8]:


plt.figure(5);
plt.plot(Iapp0,maxV,label="maxV")
plt.plot(Iapp0,minV,label="minV")
plt.plot(It,v,'b')
plt.legend()
plt.xlabel('Iapp (\mu A/cm^2)') 
plt.ylabel('voltage (mV)')

plt.figure(6);
plt.plot(tvec,gEs,"b",label="gE")
plt.plot(tvec,gIs,"r",label="gI")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('conductance (mS/cm^2)')


# In[14]:


'''
Application of 2x2 FMSD to the simulated voltage data 
'''


# In[12]:


x0n = [0,0,v[0],0,0,0]
xl = [0 for i in range(N)]

v_est = [0 for i in range(N)]
vdot_est = [0 for i in range(N)]
w_obs = [0 for i in range(N)]

for i in range(N): 
    print("(1)x0n: %s" % (str(x0n)))
    xl[i] = x0n
    dxn = fsmd_vf_22_w(x0n,v[i])
    k=0
    for j in range(6):
        x0n[j] = xl[i][j] + h*dxn[j]
    print("(2)x0n: %s" % (str(x0n)))   
    v_est[i]=x0n[2]
    vdot_est[i]=x0n[3]
    w_obs[i]=x0n[5]
    


# In[13]:


plt.figure(7);
plt.plot(tvec,vdot,"b",label="vdot")
plt.plot(tvec,vdot_est,"r",label="vdot_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('dv/dt')


plt.figure(9);
plt.plot(tvec,v,"b",label="v")
plt.plot(tvec,v_est,"r",label="v_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(10)
plt.plot(tvec,w,"b",label="w")
plt.plot(tvec,w_obs,"r",label="w_obs")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')


# In[14]:


'''
Estimation of the synaptic input after application of the filter
'''


# In[15]:


Isyn_est = [0 for i in range(N)]
Cm=20

for i in range(N):
    dx = f_g([v_est[i],w_obs[i]],Iapp)
    Isyn_est[i] =  Cm*vdot_est[i] - dx[0]     

#for i in Isyn_est:
    
error = [Isyn[i] - Isyn_est[i] for i in range(N)]
            


# In[16]:


plt.figure(11)
plt.plot(tvec,Isyn,"r",label="Isyn")
#plt.plot(tvec,Isyn_est,"g",label="Isyn_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(12)
plt.plot(tvec,Isyn_est,"b",label="Isyn_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(13)
plt.plot(tvec,Isyn,"r",label="Isyn")
plt.plot(tvec,Isyn_est,"b",label="Isyn_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(14)
plt.plot(tvec,error,"b",label="error")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')


# In[12]:


Isyn_est = [0 for i in range(N)]
Cm= 20

v_EI = [0 for i in range(N)]
w_EI = [0 for i in range(N)]
x0 = [-50,0.1]
v_EI[0]=x0[0]
w_EI[0]=x0[1]

for i in range(1,N):
    dx=f_g(x0,Iapp)
    w_EI[i] = w[i-1] + h*dx[1]
    Isyn_est[i] = Cm*vdot_est[i] - dx[0]
    x0[0]= v_est[i]
    x0[1]= w_EI[i]
    
error = [Isyn[i] - Isyn_est[i] for i in range(N)]


# In[13]:


plt.figure(15)
plt.plot(tvec,Isyn,"r",label="Isyn")
#plt.plot(tvec,Isyn_est,"g",label="Isyn_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(16)
plt.plot(tvec,Isyn_est,"b",label="Isyn_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(17)
plt.plot(tvec,Isyn,"r",label="Isyn")
plt.plot(tvec,Isyn_est,"b",label="Isyn_est")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(18)
plt.plot(tvec[50:len(error)-5],error[50:len(error)-5],"b",label="error")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')


# In[14]:


'''
Proof of concept simulation using a constant set of gEs and gIs to generate V and Ws of the ML system,
consequently, the FMSD is applied to extract the estimation of the V,Vdot,Vdoubledot and Wdot
'''


# In[22]:


h_test=0.01
tmax_test=40
N_test = int(tmax_test/h_test)

gE_av = sum(gE)/N
gI_av = sum(gI)/N

gE_test= [gE_av for i in range(N_test)]
gI_test= [gI_av for i in range(N_test)]
v_test= [0 for i in range(N_test)]
w_test= [0 for i in range(N_test)]
vdot_test = [0 for i in range(N_test)]
Isyn_test= [0 for i in range(N_test)]

v_test[0] = -50
w_test[0] = 0.1

for i in range(1,N_test):
    Isyn_test[i-1]=gE_test[i-1]*(vE-v_test[i-1])+gI_test[i-1]*(vI-v_test[i-1])
    x0=[v_test[i-1],w_test[i-1],Isyn_test[i-1]]
    dx=ML_vf(x0,Iapp)
    v_test[i] = v_test[i-1] + h*dx[0]
    w_test[i] = w_test[i-1] + h*dx[1]
    vdot_test[i] = dx[0]
    
x0n_test = [0,0,v[0],0,0,0]
xl_test = [0 for i in range(N_test)]

v_est_test = [0 for i in range(N_test)]
vdot_est_test = [0 for i in range(N_test)]
w_obs_test = [0 for i in range(N_test)]
vdotdot_est_test = [0 for i in range(N_test)]

for i in range(N_test): 
    #print("(1)x0n: %s" % (str(x0n_test)))
    xl_test[i] = x0n_test
    dxn = fsmd_vf_22_w(x0n_test,v_test[i])
    k=0
    for j in range(6):
        x0n_test[j] = xl_test[i][j] + h_test*dxn[j]
    #print("(2)x0n: %s" % (str(x0n_test)))   
    v_est_test[i]=x0n_test[2]
    vdot_est_test[i]=x0n_test[3]
    vdotdot_est_test[i]=x0n_test[4]
    w_obs_test[i]=x0n_test[5]
    
error1 = [vdot_test[i] - vdot_est_test[i] for i in range(N_test)]
error2 = [v_test[i] - v_est_test[i] for i in range(N_test)]
error3 = [w_test[i] - w_obs_test[i] for i in range(N_test)]

print(w_test)
#print(error1)
#print(error2)


# In[23]:


'''
Estimation of the gEs and gIs
'''


# In[24]:


Cm=20.0
phi=0.066667
    
gCa=4.0
gL=2.0
gK=8.0
    
vCa=120.0
vL=-60.0
vK=-84.0
    
v1=-1.2
v2=18
v3=12
v4=17.4

gE_test_est = [0 for i in range(N_test)]
gI_test_est = [0 for i in range(N_test)]
x=[0,0]
k = [0 for i in range(N_test)]
gE_est_test = [0 for i in range(N_test)]
gI_est_test = [0 for i in range(N_test)]

for i in range(N_test):
    print("iteration %d" % (i))
    iL=2.0*(v_est_test[i]+60.0)
    print("iL %.6f" % iL)
    minf=0.5*(1+np.tanh((v_est_test[i]+1.2)/18.0))
    print("minf %.6f" % minf)
    iCa=4.0*minf*(v_est_test[i]-120.0)
    print("iCa %.6f " % iCa)
    iK=8.0*w_test[i]*(v_est_test[i]+84.0)
    print("iK %.6f" %iK)
    
    print("v: %.2f, w: %.2f" % (v_est_test[i],w_test[i]))
    
    #Until here all computations match analytical numbers 
    
    minf_dot= 0.5*18*(1/(np.cosh((v_est_test[i]+1.2)/18)*np.cosh((v_est_test[i]+1.2)/18)))
    #note that df_v and df_w have negative multipliers but due to the final algebraic expression they are ommitted
    df_v = (gL + gCa*minf_dot*(v_est_test[i]-vCa) + gCa*minf + gK*w_test[i])/Cm
    df_w = gK*(v_est_test[i]-vK)/Cm
    
    print("minf_dot: %.6f, df_v: %.6f, df_w: %.6f" % (minf_dot,df_v,df_w))

    c1 = vdot_est_test[i] + (iL + iCa + iK)/Cm
    c2 = vdotdot_est_test[i] + (vdot_est_test[i]*df_v + w_test[i]*df_w)/Cm
    
    print("vdot: %.6f, vdotdot: %.6f" % (vdot_est_test[i],vdotdot_est_test[i]))
    
    c2s = c2*Cm/-vdot_est_test[i]
    
    a1 = [(vE-v_est_test[i])/Cm,(vI-v_est_test[i])/Cm]
    a2 = [1,1]
    a = np.array([a1,a2])
    b = np.array([c1,c2s])
    x = np.linalg.solve(a,b)
    #k[i]= np.linalg.lstsq(a,b)
    #print(k[i])
    gE_est_test[i] = x[0]/20
    gI_est_test[i] = x[1]/20

'''
for i in range(N_test):
    gE_est_test[i] = k[i][0][0]
    gI_est_test[i] = k[i][0][1]
'''

print(gE_est_test)

tvec_test = np.arange(0,tmax_test,h_test)

plt.figure(20)
plt.plot(tvec_test,v_test,"r",label="v_test")
plt.plot(tvec_test,v_est_test,"b",label="v_est_test")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')

plt.figure(21)
plt.plot(tvec_test,gE_est_test,"r",label="gE")
plt.plot(tvec_test,gI_est_test,"b",label="gI")
plt.legend()
plt.xlabel('time (ms)') 
plt.ylabel('voltage(mV)')
    
    


# In[23]:


gE_av


# In[24]:


gI_av


# In[37]:


(1 + np.tanh(-48.8/18))*0.5


# In[43]:


0.5*(1+np.tanh((-50+1.2)/18.0))


# In[25]:


0.5*18*(1/(np.cosh(-48.2/18)*np.cosh(-48.2/18)))

