import matplotlib
import sys 
sys.path.append("path there I store the Hamiltonian")
import dataforadapt 
import numpy as np
import matplotlib.pyplot as plt
import copy
import functools as ft
from scipy.optimize import minimize
import itertools
import scipy
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import config
config.update("jax_enable_x64", True)
def szall(N):                            #generate sz single operators ZI....I,IZI...I,IIZI...I,...,I...IZI,I....IZ
    szall=[[] for i in range(N)]
    szoper=np.array([[1,0],[0,-1]],dtype=complex)
    tensorized=[]
    for j in range(N):
     for i in range(N):
       if i==j:
           szall[j].append(szoper)
       else:
           szall[j].append(np.eye(2))
    for i in range(N):
        tensorized.append(ft.reduce(np.kron,szall[i]))
    return tensorized


def sxall(N):            #generate sx single operators XI....I,IXI...I,IIXI...I,...,I...IXI,I....IX
    sxall=[[] for i in range(N)]
    sxoper=np.array([[0,1],[1,0]],dtype=complex)
    tensorized=[]
    for j in range(N):
     for i in range(N):
       if i==j:
           sxall[j].append(sxoper)
       else:
           sxall[j].append(np.eye(2))
    for i in range(N):
        tensorized.append(ft.reduce(np.kron,sxall[i]))
    return tensorized


def syall(N):                       #generate sy single operators YI....I,IYI...I,IIYI...I,...,I...IYI,I....IY
    syall=[[] for i in range(N)]
    syoper=np.array([[0,-1j],[1j,0]])
    tensorized=[]
    for j in range(N):
     for i in range(N):
       if i==j:
           syall[j].append(syoper)
       else:
           syall[j].append(np.eye(2))
    for i in range(N):
        tensorized.append(ft.reduce(np.kron,syall[i]))
    return tensorized

def initial(array):         #initial state(HF solution)
    basisstate=[]
    tensorized=[]
    for i in range(np.shape(array)[0]):
        if array[i]==0:
            basisstate.append([[1.0],[0.0]])
        elif array[i]==1:
            basisstate.append([[0.0],[1.0]])
        else:
            pass
    tensorized.append(ft.reduce(np.kron,basisstate))    
    return np.array((tensorized[0]).flatten()) 

def expm1(a,th,n):                                  #faster exponential for pauli operators
   return jnp.cos(th)*jnp.eye(2**n,dtype=complex)-1j*jnp.sin(th)*a


def pool(N,sx,sy,sz):                         #pool of operators that help to get the ground state fast
   po=[]
   pa=[]
   for i in range(N):
    for j in range(N):
      if (i+j)%2==0: 
       po.append(sx[i].dot(sy[j]))
       pa.append('X_{}Y_{}'.format(i,j))
      #print(i)    
   for i in range(N):
    for j in range(N):
     for k in range(N):
      for l in range(N): 
        if (i+j+k+l)%2==0 and i!=j and i!=k and i!=l and j!=k and j!=l:
         po.append(np.linalg.multi_dot([sx[i],sy[j],sy[k],sy[l]])) #that's give correct 2.1651
         pa.append('X_{}Y_{}Y_{}Y_{}'.format(i,j,k,l))

   return po,pa

def allpauli(n):                           #a whole space of 4^n Pauli operators for n-qubit space
   ide=np.eye(2)
   sx1=np.array([[0,1],[1,0]])
   sy1=np.array([[0,-1j],[1j,0]])
   sz1=np.array([[1,0],[0,-1]])
   axx=[sx1,sy1,sz1]
   ia=['x','y','z']
   y1=list(itertools.product(axx,repeat=n))
   pstringpau=list(itertools.product(ia,repeat=n))
   yall=[ft.reduce(np.kron,y1[i]) for i in range(len(y1))]

   return yall,pstringpau

N=8
sxop=sxall(N)
syop=syall(N)
szop=szall(N)
h=dataforadapt.h4closedfermion()
w,e=np.linalg.eigh(h)
ins=initial([1,0,0,1,1,0,0,1])
ps,pastring=pool(N,sxop,syop,szop)

def uall(yx,psall):                     #apply operators one by one.
   uop=[]
   for ia in reversed(range(len(yx))):
      uop.append(expm1(psall[ia],-yx[ia],N))
   if len(yx)==1:
      y=uop[0]
   else:
      y=jnp.linalg.multi_dot(uop)
   return y

def gradient(psaa,isa,hami):           #calculate all gradients for operators
 grall=[]
 for i in range(len(psaa)):
   hc=hami.dot(psaa[i])-psaa[i].dot(hami)
   grall.append(np.vdot(isa,hc.dot(isa))) 
 #vva=[]
 #bv1=sorted(map(abs,grall))
 #k=1
 #for j in range(len(grall)):
 #     if np.allclose(grall[j],bv1[-1]):  
 #       vva.append(j)
 #print(vv)       
 #if len(vva)>0:
 # for i in range(len(vva)):
 #   del grall[vva[i]]
 #   del ps[vva[i]]      
 return grall  

def maxgr(psbv,grelements):          #choose max gradient
 maxall=-1
 for i in range(len(grelements)):
   if abs(grelements[i])>maxall:
      maxall=abs(grelements[i])
      ind=i
 return maxall,ind,psbv[ind]


psalla=[]
indall=[]
asf=100   #initial random value for the energy
kit=0    # of iterations
enall=[]
fid=[]
ende=[]
psitel2=ins
xparameters=[0]
opeforadapt=[]
while asf-w[0]>10**(-7):
 if kit==0:               
  grss=gradient(ps,ins,h)
  max1,ind1,ps1=maxgr(ps,grss)
  psalla.append(ps1)
  indall.append(ind1)
 else:
    if len(psalla)==1: 
     grss=gradient(ps,expm1(psalla[0],-xparameters[0],N).dot(ins),h)
     max1,ind1,ps1=maxgr(ps,grss)
     psalla.append(ps1)
     indall.append(ind1)
    else:
       grss=gradient(ps,jnp.linalg.multi_dot([expm1(psalla[i],-xparameters[i],N) for i in reversed(range(len(psalla)))]).dot(ins),h)
       max1,ind1,ps1=maxgr(ps,grss)
       psalla.append(ps1)
       indall.append(ind1) 
 def fun(x):
   global psalla
   u1=uall([x[i] for i in range(len(psalla))],psalla)
   psitel=jnp.dot(u1,ins)
   return jnp.real(jnp.vdot(psitel,jnp.dot(h,psitel)))
 obj_and_grad = jit(value_and_grad(fun))
 if kit==0:
    xa=[0]
 else:
    xa=[]
    b=[]
    for ka in range(len(xparameters)):
       xa.append(xparameters[ka])
    xa.append(0)   
 print(np.var(grss),pastring[ind1],ind1)
 with open("Path for storing operators that contribute the ground state preparation'", "a") as text_file:
    text_file.write("%s" % pastring[ind1]+ "\n")
 opeforadapt.append(pastring[ind1])
 yall1=minimize(obj_and_grad,x0=xa,method='L-BFGS-B',options={'gtol':10**(-15)},jac=True) #check different optimizers depending on the problem, you may get better results
 asf=yall1.fun
 xparameters=yall1.x
 u1=uall([xparameters[i] for i in range(len(psalla))],psalla)
 psitel2=u1.dot(ins)
 psitel2=psitel2/jnp.linalg.norm(psitel2)
 print(asf,w[0],asf-w[0],kit,1-abs(np.vdot(e[:,0],psitel2))**2)
 print(xparameters)
 kit+=1
 ende.append(abs(asf-w[0]))
 fid.append(1-abs(np.vdot(e[:,0],psitel2))**2)
 enall.append(asf)
np.savetxt('Path for storing parameters of the ground state preparations', xparameters, delimiter =',') 

fig=plt.figure() 
ax=plt.axes()
ax.scatter(range(kit),ende,color='g')
ax.set_xlabel('iterations')
ax.set_yscale('log')
ax.set_ylabel('ΔΕ')

fig2=plt.figure() 
ax2=plt.axes()
ax2.scatter(range(kit),fid,color='b')
ax2.set_xlabel('iterations')
ax2.set_yscale('log')
ax2.set_ylabel('Infidelity')



plt.show()  
