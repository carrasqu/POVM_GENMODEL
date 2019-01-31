import numpy as np
from ncon import ncon
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state
from readMPS import readMPS


class Fid():
    def __init__(self, POVM='Trine',Number_qubits=4,MPS='GHZ',Nmax=10000):


        self.N = Number_qubits;

        # POVMs and other operators
        # Pauli matrices
        self.I = np.array([[1, 0],[0, 1]]);
        self.X = np.array([[0, 1],[1, 0]]);    self.s1 = self.X;
        self.Z = np.array([[1, 0],[0, -1]]);   self.s3 = self.Z;
        self.Y = np.array([[0, -1j],[1j, 0]]); self.s2 = self.Y;
        self.Nmax = Nmax

        # Which POVM 
        if POVM=='4Pauli':
            self.K = 4;

            self.M = np.zeros((self.K,2,2),dtype=complex);

            self.M[0,:,:] = 1.0/3.0*np.array([[1, 0],[0, 0]])
            self.M[1,:,:] = 1.0/6.0*np.array([[1, 1],[1, 1]])
            self.M[2,:,:] = 1.0/6.0*np.array([[1, -1j],[1j, 1]])
            self.M[3,:,:] = 1.0/3.0*(np.array([[0, 0],[0, 1]]) + \
                                     0.5*np.array([[1, -1],[-1, 1]]) \
                                   + 0.5*np.array([[1, 1j],[-1j, 1]]) )
 
        if POVM=='Tetra':
            self.K=4;

            self.M=np.zeros((self.K,2,2),dtype=complex);

            self.v1=np.array([0, 0, 1.0]);
            self.M[0,:,:]=1.0/4.0*( self.I + self.v1[0]*self.s1+self.v1[1]*self.s2+self.v1[2]*self.s3);

            self.v2=np.array([2.0*np.sqrt(2.0)/3.0, 0.0, -1.0/3.0 ]);
            self.M[1,:,:]=1.0/4.0*( self.I + self.v2[0]*self.s1+self.v2[1]*self.s2+self.v2[2]*self.s3);

            self.v3=np.array([-np.sqrt(2.0)/3.0 ,np.sqrt(2.0/3.0), -1.0/3.0 ]);
            self.M[2,:,:]=1.0/4.0*( self.I + self.v3[0]*self.s1+self.v3[1]*self.s2+self.v3[2]*self.s3);

            self.v4=np.array([-np.sqrt(2.0)/3.0, -np.sqrt(2.0/3.0), -1.0/3.0 ]);
            self.M[3,:,:]=1.0/4.0*( self.I + self.v4[0]*self.s1+self.v4[1]*self.s2+self.v4[2]*self.s3);

        elif POVM=='Trine':
            self.K=3;
            self.M=np.zeros((self.K,2,2),dtype=complex);
            phi0=0.0
            for k in range(self.K):
                phi =  phi0+ (k)*2*np.pi/3.0
                self.M[k,:,:]=0.5*( self.I + np.cos(phi)*self.Z + np.sin(phi)*self.X)*2/3.0

        #% T matrix and its inverse
        self.t = ncon((self.M,self.M),([-1,1,2],[ -2,2,1]));
        self.it = np.linalg.inv(self.t);
        # Tensor for expectation value
        self.Trsx  = np.zeros((self.N,self.K),dtype=complex);
        self.Trsy  = np.zeros((self.N,self.K),dtype=complex);
        self.Trsz  = np.zeros((self.N,self.K),dtype=complex);
        self.Trrho = np.zeros((self.N,self.K),dtype=complex);
        self.Trrho2 = np.zeros((self.N,self.K,self.K),dtype=complex);
        self.T2 = np.zeros((self.N,self.K,self.K),dtype=complex);
 

 
        if MPS=="GHZ":
            # Copy tensors used to construct GHZ as an MPS. The procedure below should work for any other MPS 
            cc = np.zeros((2,2)); # corner
            cc[0,0] = 2**(-1.0/(2*self.N));
            cc[1,1] = 2**(-1.0/(2*self.N));
            cb = np.zeros((2,2,2)); # bulk
            cb[0,0,0] = 2**(-1.0/(2*self.N));
            cb[1,1,1] = 2**(-1.0/(2*self.N));
        
       
            self.MPS = []
            self.MPS.append(cc)
            for i in range(self.N-2):
                self.MPS.append(cb)
            self.MPS.append(cc) 
        else:
             # Read MPS from files 
             self.MPS = readMPS(MPSf=MPS, N=self.N,convert=True)
        

    def Fidelity(self,S):
        Fidelity = 0.0;
        F2 = 0.0;
        Ns = S.shape[0]
        for i in range(Ns):

            # contracting the entire TN for each sample S[i,:]
            eT = ncon((self.it[:,S[i,0]],self.M,self.MPS[0],self.MPS[0]),([3],[3,2,1],[1,-1],[2,-2]));

            for j in range(1,self.N-1):
                eT = ncon((eT,self.it[:,S[i,j]],self.M,self.MPS[j],self.MPS[j]),([2,4],[1],[1,5,3],[2,3,-1],[4,5,-2]));

            
            j = self.N-1
            eT = ncon((eT,self.it[:,S[i,j]],self.M,self.MPS[j],self.MPS[j]),([2,5],[1],[1,4,3],[3,2],[4,5]));
            Fidelity = Fidelity + eT;
            F2 = F2 + eT**2;
            Fest=Fidelity/float(i+1);
            F2est=F2/float(i+1);
            Error = np.sqrt( np.abs( F2est-Fest**2 )/float(i+1));
            
        F2 = F2/float(Ns);

        Fidelity = np.abs(Fidelity/float(Ns));

        Error = np.sqrt( np.abs( F2-Fidelity**2 )/float(Ns));

        return np.real(Fidelity), Error

    def cFidelity(self,S,logP):
        Fidelity = 0.0;
        F2 = 0.0;
        Ns = S.shape[0]
        if Ns > self.Nmax:
           Ns = self.Nmax
        KL = 0.0
        K2 = 0.0 
        for i in range(Ns):
            
            P = ncon(( self.MPS[0], self.MPS[0],self.M[S[i,0],:,:]),([1,-1],[2,-2],[1,2]))  
             

            # contracting the entire TN for each sample S[i,:]  
            for j in range(1,self.N-1):
                P = ncon((P,self.MPS[j], self.MPS[j],self.M[S[i,j],:,:]),([1,2],[1,3,-1],[2,4,-2],[3,4]))

            P = ncon((P,self.MPS[self.N-1], self.MPS[self.N-1],self.M[S[i,self.N-1],:,:]),([1,2],[3,1],[4,2],[3,4]))

            ee = np.sqrt(P/np.exp(logP[i]))
            Fidelity = Fidelity + ee
            F2 = F2 + ee**2

            KL = KL + 2*np.log(ee);
            K2 = K2 +4*(np.log(ee))**2;
            Fe = Fidelity/float(i+1)
            #Fe2 = F2/float(i+1)
            #Fe2 = np.sqrt(np.abs(  Fe**2-Fe2 )/float(i+1))
 
            #print  Fe,Fe2   

        F2 = F2/float(Ns);

        Fidelity = np.abs(Fidelity/float(Ns));

        Error = np.sqrt( np.abs( F2-Fidelity**2 )/float(Ns));

        K2 = K2/float(Ns);

        KL = np.abs(KL/float(Ns));

        ErrorKL = np.sqrt( np.abs( K2-KL**2 )/float(Ns));


        return np.real(Fidelity), Error, np.real(KL), ErrorKL



