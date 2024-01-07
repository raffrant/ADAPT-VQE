import matplotlib
import sys 
sys.path.append("C:/Users/rfrantzesk/Desktop/physics/qisadapt")
import dataforadapt 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import functools as ft
from scipy.optimize import minimize,differential_evolution
import itertools
import random
from matplotlib.collections import LineCollection
import sympy
def szall(N):
    szall=[[] for i in range(N)]
    szoper=np.array([[1,0],[0,-1]],dtype=complex)
    #j=0
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

def expm1(a,th,n):
   return np.cos(th/2)*np.eye(2**n,dtype=complex)+1j*np.sin(th/2)*a
def sxall(N):
    sxall=[[] for i in range(N)]
    sxoper=np.array([[0,1],[1,0]],dtype=complex)
    #j=0
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


def syall(N):
    syall=[[] for i in range(N)]
    syoper=np.array([[0,-1j],[1j,0]])
    #j=0
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
    
    
def iniplus(N):
      tensorized=[]
      for i in range(N):
        tensorized.append([[1.0/np.sqrt(2)],[1.0/np.sqrt(2)]])
      return ft.reduce(np.kron,tensorized).flatten()
def initial(array):
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

def egh(op,op1,theta):
   a1=[]
   for i in range(len(op)):
      a1.append(np.matmul(expm1(op1[i],theta[i],N),np.matmul(op,expm1(op1[i],-theta[i],N))))

   return a1   
def swap(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)

pool=[]
thetaarx=np.random.uniform(low=0.2,high=1.2,size=len(pool))

N=8
sxop=sxall(N)
syop=syall(N)
szop=szall(N)

h=dataforadapt.ham()


w,e=np.linalg.eigh(h)

def pool(N,sx,sy,sz):
   po=[]
   pa=[]
   for i in range(N):
    for j in range(N):
      if (i+j)%2 ==0 and i!=j and i<j:  
       po.append(sx[i].dot(sy[j]))
       po.append(sy[i].dot(sx[j]))
       pa.append('X_{}Y_{}'.format(i,j))
       pa.append('Y_{}X_{}'.format(i,j))
      #print(i)    
   for i in range(N):
    for j in range(N):
     for k in range(N):
      for l in range(N): 
        if (i+j+k+l)%2==0 and i!=j and i!=k and i!=l and j!=k and j!=l:
        # po.append(np.linalg.multi_dot([sx[i],sx[j],sx[k],sy[l]]))
        # po.append(np.linalg.multi_dot([sy[i],sx[j],sx[k],sx[l]]))
         #po.append(np.linalg.multi_dot([sx[i],sy[j],sx[k],sx[l]]))
       #  po.append(np.linalg.multi_dot([sx[i],sx[j],sy[k],sx[l]]))
         po.append(np.linalg.multi_dot([sy[i],sy[j],sy[k],sx[l]])) #that's give correct 2.1651
        # po.append(np.linalg.multi_dot([sx[i],sy[j],sy[k],sy[l]]))
         #po.append(np.linalg.multi_dot([sy[i],sx[j],sy[k],sy[l]]))
         #po.append(np.linalg.multi_dot([sy[i],sy[j],sx[k],sy[l]]))
         #pa.append('X_{}X_{}X_{}Y_{}'.format(i,j,k,l))
         #pa.append('Y_{}X_{}X_{}X_{}'.format(i,j,k,l))
         #pa.append('X_{}Y_{}X_{}X_{}'.format(i,j,k,l))
        # pa.append('X_{}X_{}Y_{}X_{}'.format(i,j,k,l))
         pa.append('Y_{}Y_{}Y_{}X_{}'.format(i,j,k,l))
         #pa.append('X_{}Y_{}Y_{}Y_{}'.format(i,j,k,l))
        # pa.append('Y_{}X_{}Y_{}Y_{}'.format(i,j,k,l))
       #  pa.append('Y_{}Y_{}X_{}Y_{}'.format(i,j,k,l))
    

   return po,pa

def allpauli(n):
   ide=np.eye(2)
   sx1=np.array([[0,1],[1,0]])
   sy1=np.array([[0,-1j],[1j,0]])
   sz1=np.array([[1,0],[0,-1]])
   axx=[ide,sx1,sy1,sz1]
   tell=[axx,axx,axx,axx]
   y1=list(itertools.product(axx,repeat=n))
   #stringlist=list(itertools.product(['i','x','y','z'],repeat=n))
   yall=[ft.reduce(np.kron,y1[i]) for i in range(4**n)]
   for i in range(len(yall)):
      yall[i]=1j*yall[i]
   return yall
#ins=np.kron(initial([random.randrange(2) for _ in range(N-1)]),np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]])).flatten()
ins=initial([1,1,1,1,0,0,0,0])

#print(ins,e[:,0])
#ft.reduce(np.kron,[np.array([[0],[1]]),iniplus(N-4),np.array([[0],[1]]),np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]]),np.array([[1],[0]])]).flatten()#initial(np.zeros(N))
#

ps,pastring=dataforadapt.poh4()#pool(N,sxall(N),syall(N),szall(N))
#print(len(pastring))

def uall(yx,psall):
   uop=[]
   uop2=[]
   for i in range(len(yx)):
      uop.append(expm1(psall[i],-yx[i],N))
      uop2.append(expm1(psall[i],yx[i],N))
   if len(uop)==1:
      y=  uop[0]
      y1= uop2[0]
   else:
      y=  np.linalg.multi_dot(uop)
      y1= np.linalg.multi_dot(uop2)
   return y,y1

def gradient(ps,isa):
 grall=[]
 for i in range(len(ps)):
   hc=np.matmul(h,ps[i])-np.matmul(ps[i],h)
   grall.append(np.vdot(isa,hc.dot(isa)))
 vv=[]
 bv1=sorted(map(abs,grall))
 k=1
 for j in range(len(grall)):
     if abs(bv1[-1]-abs(grall[j]))<0.01*abs(bv1[-1]):
        vv.append(j)
 print(vv)       
 if len(vv)>0:
  for i in range(len(vv)):
    #del grall[vv[i]]
    del ps[vv[i]]
 return grall  
def maxgr(grelements):
 global ps
 maxall=-1
 for i in range(len(grelements)):
   if abs(grelements[i])>maxall:
      maxall=abs(grelements[i])
      ind=i
 return maxall,ind,ps[ind]
psalla=[]
indall=[]
asf=100
k=0
kit=0
enall=[]
fid=[]
ende=[]
psitel2=ins
xparameters=[0]
while asf-w[0]>10**(-12) and kit<30:
 if k==0:
  grss=gradient(ps,ins)
  max1,ind1,ps1=maxgr(grss)
  psalla.append(ps1)
  indall.append(ind1)
 else:
    if len(psalla)==1: 
     grss=gradient(ps,expm1(psalla[0],-xparameters[0],N).dot(ins))
     max1,ind1,ps1=maxgr(grss)
     psalla.append(ps1)
     indall.append(ind1)
    else:
       grss=gradient(ps,np.linalg.multi_dot([expm1(psalla[i],-xparameters[i],N) for i in range(len(psalla))]).dot(ins))
       max1,ind1,ps1=maxgr(grss)
       psalla.append(ps1)
       indall.append(ind1) 
      
 def fun(x):
   global psalla
   u1,u2=uall([x[i] for i in range(len(psalla))],psalla)
   psitel=(u2.dot(h.dot(u1))).dot(ins)
   #
   #psitel2=psitel2/np.linalg.norm(psitel2)
   #print(np.vdot(e[:,0],psitel2))
   return np.real(np.vdot(ins,psitel))
 k+=1
 if k==1:
    xa=[0]
    b=[(0,2*np.pi)]
   # xsy=[sympy.Symbol('x_0')]
 else:
    xa=[]
    b=[]
    #xsy=[]
    for ka in range(len(xparameters)):
       xa.append(xparameters[ka])
       b.append((0,2*np.pi))
    #   xsy.append(sympy.Symbol('x_{}'.format(ka)))
    xa.append(0)   
    b.append((0,2*np.pi))
   # xsy.append(sympy.Symbol('x_{}'.format(ka+1)))
   # print(xsy)
 #pexp=sympy.exp(u1)   
 print(np.var(grss),ind1,pastring[ind1],len(ps))
 #com=sympy.Matrix(h)*sympy.Matrix(ps[ind1])-sympy.Matrix(ps[ind1])*sympy.Matrix(h)
 #allmatrix1=sympy.eye(2**N)
 #allmatrix2=sympy.eye(2**N)
 #for i in range(len(xparameters)):
 #   allmatrix1=allmatrix1*(-xsy[i])*sympy.Matrix(ps[indall[i]])
 #   allmatrix2=allmatrix2*(xsy[i])*sympy.Matrix(ps[indall[i]])
 #expr=(sympy.Matrix(psitel2).conjugate()).dot(allmatrix2*com*allmatrix1*sympy.Matrix(psitel2))
 #bv=set()
 #baa=set()
 #for ij in range(len(xparameters)):
 #   bv.add(xsy[ij])
 #   baa.add(expr.diff(xsy[i]))
 #print(sympy.simplify(xsy))   
 #jac = sympy.lambdify((bv), sympy.Array([baa]),"scipy")
 #yall1=differential_evolution(fun,x0=xa,bounds=[(0,2*np.pi) for i in range(len(xa))],maxiter=100,tol=10**(-10))
 yall1=minimize(fun, x0=xa,method="BFGS",tol=10**(-20))#,options={'maxiter':500})#,bounds=b)#,jac=jac)#gradient(psitel2,ind1))   
 asf=yall1.fun
 xparameters=yall1.x
 print(len(xparameters))
 u1,u2=uall([xparameters[i] for i in range(len(psalla))],psalla)
 psitel2=u1.dot(ins)
 print(asf,w[0],abs(asf-w[0]),kit)
 kit+=1
 fid.append(abs(np.vdot(e[:,0],psitel2))**2)
 #print(xparameters)
 ende.append(abs(asf-w[0]))
 enall.append(asf)
#plt.plot(range(k),abs(enall-w[0]*np.ones(k))/abs(w[0]))
plt.scatter(range(kit),ende)
#plt.yscale('log')
plt.xlabel('iterations')
plt.yscale('log')
plt.ylabel('ΔΕ')
plt.show()

'''
p1=[]
p2=[]
p3=[]
p4=[]
pall=[]
for i in range(1,N):
   p1.append("x0yj".replace('j', '{}'.format(i)))
   p2.append("y0xj".replace('j', '{}'.format(i)))
   p3.append("x0xjx0yj".replace('j', '{}'.format(i)))
   p4.append("y0xjx0xj".replace('j', '{}'.format(i)))
   #p2.append(np.matmul(syop[0],sxop[i]))
   #p3.append()
   pall.append(p1[i-1])
   pall.append(p2[i-1])
p1correct=[]
p2correct=[]
for i in range(len(p1)):
   if p1[i][0]=='x' :
    p1correct.append(np.matmul(sxop[0],syop[int(p1[i][3])]))   
    print(True)
   if p2[i][0]=='y' :
    p2correct.append(np.matmul(syop[0],sxop[int(p2[i][3])]))
pall31=[]
pall41=[]
k=0
for j in range(len(p3)):
 for i in range(0,len(p3[j]),2):
   pall31.append(swap(p3[j],6,i))
   pall.append(pall31[k])
   k+=1
k=0   
for j in range(len(p4)):
 for i in range(0,len(p4[j]),2):
   pall41.append(swap(p4[j],6,i))   
   pall.append(pall31[k])
   k+=1
   #print(i)

pop=[]
for i in range(len(pall)):
       if len(pall[i])==4:   
         if pall[i][0]=='x' :
          pop.append(np.matmul(sxop[0],syop[int(pall[i][3])]))
         if pall[i][0]=='y' :
          pop.append(np.matmul(syop[0],sxop[int(pall[i][3])])) 
       if len(pall[i])==8:   
          ind=[pall[i][j+1] for j in range(4)]
          pal=[pall[i][j] for j in range(4)]
          amm=[]
          for ij in range(len(ind)):
            if pal[ij]=='x': 
             amm.append(sxop[int(ind[ij])])
            if pal[ij]=='y': 
             amm.append(syop[int(ind[ij])])
          pop.append(np.linalg.multi_dot(amm))   
'''


