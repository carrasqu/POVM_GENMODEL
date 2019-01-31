import numpy as np
from fidelityMPS import Fid
import sys

s = sys.argv[1]
ind = sys.argv[2]
Nmax = sys.argv[3]

i=s
S=np.loadtxt("samplex_"+str(i)+".txt",dtype=int)-1
Nq=S.shape[1]

Number_qubits = S.shape[1]
Ficalc=Fid(POVM='4Pauli',Number_qubits=Nq,MPS='tensor.txt',Nmax=int(Nmax))
print("here")
for i in range(int(s),int(ind),1):


    S = np.loadtxt("samplex_"+str(i)+".txt",dtype=int)-1
    #S = np.flip(S,1) # because IBM flips the spins
 
    logP = np.loadtxt("logP_"+str(i)+".txt") 

    Nq = S.shape[1]

    Number_qubits = S.shape[1]

    F,E = Ficalc.Fidelity(S)
    cF,cE,KL,eKL = Ficalc.cFidelity(S,logP)
  
    print(i, cF,cE,KL,eKL)
