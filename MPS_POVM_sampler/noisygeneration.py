import numpy as np
from ncon import ncon
from  readMPS import readMPS
import sys
class PaMPS():
    def __init__(self, POVM='Trine',Number_qubits=4,MPS='GHZ',p=0.0):


        self.N = Number_qubits;

        # POVMs and other operators
        # Pauli matrices
        self.I = np.array([[1, 0],[0, 1]]);
        self.X = np.array([[0, 1],[1, 0]]);    self.s1 = self.X;
        self.Z = np.array([[1, 0],[0, -1]]);   self.s3 = self.Z;
        self.Y = np.array([[0, -1j],[1j, 0]]); self.s2 = self.Y;


        # Tetra POVM
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
   
        elif POVM=='Pauli':
            self.K = 6;
            Ps = np.array([1./3., 1./3., 1./3., 1./3.,1./3.,1./3.]); 
            self.M = np.zeros((self.K,2,2),dtype=complex);
            theta = np.pi/2.0
            self.M[0,:,:] = Ps[0]*self.pXp(theta,0.0)
            self.M[1,:,:] = Ps[1]*self.mXm(theta,0.0)
            self.M[2,:,:] = Ps[2]*self.pXp(theta,np.pi/2.0)
            self.M[3,:,:] = Ps[3]*self.mXm(theta,np.pi/2.0)
            self.M[4,:,:] = Ps[4]*self.pXp(0.0,0.0)
            self.M[5,:,:] = Ps[5]*self.mXm(0,0.0)
        elif POVM=='Pauli_rebit':
            self.K = 4;
            Ps = np.array([1./2., 1./2., 1./2., 1./2.]);
            self.M = np.zeros((self.K,2,2),dtype=complex);
            theta = np.pi/2.0
            self.M[0,:,:] = Ps[0]*self.pXp(theta,0.0)
            self.M[1,:,:] = Ps[1]*self.mXm(theta,0.0)
            self.M[2,:,:] = Ps[2]*self.pXp(0.0,0.0)
            self.M[3,:,:] = Ps[3]*self.mXm(0,0.0)

        elif POVM == 'Trine':
            self.K = 3;
            self.M = np.zeros((self.K,2,2),dtype=complex);
            phi0=0.0
            for k in range(self.K):
                phi =  phi0+ (k)*2*np.pi/3.0
                self.M[k,:,:] = 0.5*( self.I + np.cos(phi)*self.Z + np.sin(phi)*self.X)*2/3.0
       
        elif POVM == 'Psi2':
            # If the POVM is replaced by a copy tensor, then this code samples the square of the MPS if MPS is normalized
            self.K = 2
            self.M = np.zeros((self.K,2,2),dtype=complex); 
            self.M[0,0,0] = 1
            self.M[1,1,1] = 1    


        if MPS=='GHZ': 
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
             self.MPS = readMPS(MPSf=MPS, N=self.N,convert=True)                
 
        # constructing the MPO for the locally depolarized GHZ state from its  MPS representation
        USA=np.zeros((2,2,4,4));

        E00 = np.zeros((4,4));
        E10 = np.zeros((4,4));
        E20 = np.zeros((4,4));
        E30 = np.zeros((4,4));
        E00[0,0] = 1;
        E10[1,0] = 1;
        E20[2,0] = 1;
        E30[3,0] = 1;


        USA = USA + np.sqrt(1.0-p)*ncon((self.I,E00),([-1,-2],[-3,-4]))
        USA = USA + np.sqrt(p/3.0)*ncon((self.s1,E10),([-1,-2],[-3,-4]))
        USA = USA + np.sqrt(p/3.0)*ncon((self.s2,E20),([-1,-2],[-3,-4]))
        USA = USA + np.sqrt(p/3.0)*ncon((self.s3,E30),([-1,-2],[-3,-4]))

        E0=np.zeros((4));
        E0[0] = 1;

        self.locMixer = ncon( ( USA,E0, np.conj(USA), E0 ),([-1,-2,1,3],[3],[-4,-3,1,2],[2]));

        self.LocxM = ncon((self.M,self.locMixer),([-3,1,2],[2,-2,-1,1]))

        Tr = np.ones((self.K)); 

        self.l_P = [None] * self.N
        #self.l_P[self.N-1] = ncon((self.M,self.MPS[self.N-1],self.MPS[self.N-1],self.locMixer),([-3,1,2],[-1,3 ],[-2,4],[2,4,3,1])); ### BUGGGGG aaaaaarrg
        self.l_P[self.N-1] = ncon((self.M,self.MPS[self.N-1],self.MPS[self.N-1],self.locMixer),([-3,1,2],[3,-1 ],[4,-2],[2,4,3,1]));

        for i in range(self.N-2,0,-1):
            self.l_P[i] = ncon((self.M  , self.MPS[i], self.MPS[i],self.locMixer,self.l_P[i+1],Tr),\
                               ([-3,4,5], [-1,6,2],    [-2,7,3],   [5,7,6,4],    [2,3,1 ],  [1]));

        self.l_P[0] = ncon((self.M,   self.MPS[0], self.MPS[0], self.locMixer, self.l_P[1], Tr),\
                           ([-1,4,5 ], [6,2],      [7,3],       [5,7,6,4],     [2,3,1 ],    [1]));
 
        return      
 
    def samples(self, Ns=1000000, fname='train.txt'):
        
        f = open(fname, 'w')
        f2 = open("data.txt", 'w')

        state = np.zeros((self.N),dtype=np.uint8);
        for  ii in range(Ns):
            
            Pi = np.real(self.l_P[0])  #ncon((self.TrP[1],self.l_P[0]),([1,2],[1,2,-1]));
            Pnum = Pi;
            #print Pi,np.sum(Pi)
             
            
            i=1;

            #state[0] = np.random.choice(self.K, 1, p=Pi) #andsample(1:K,1,true,Pi);
            state[0] = (np.argmax(np.random.multinomial(n=1, pvals=Pi,size=1)))
   
            Pden = Pnum[state[0]];

            PP = ncon((self.M[state[0]], self.locMixer, self.MPS[0], self.MPS[0] ),\
                          ([2,1],            [1,4,3,2],       [3,-1],      [4,-2]))    


            for i in range(1,self.N-1):  #=2:N-1

                Pnum = np.real(ncon((PP,self.l_P[i]),([1,2],[1,2,-1])));
                Pi   = Pnum/Pden;
                state[i] =  np.argmax(np.random.multinomial(n=1, pvals=Pi,size=1)) #np.random.choice(self.K, 1, p=Pi) #randsample(1:K,1,true,Pi);
                Pden = Pnum[state[i]];
                PP =  ncon((PP,self.LocxM[:,:,state[i]] ,self.MPS[i],self.MPS[i] ),([1,2],[3,4],[1,3,-1],[2,4,-2]));
              
            i = self.N-1;
            Pnum = np.real(ncon((PP,self.l_P[self.N-1]),([1,2],[1,2,-1])));
            Pi   =   Pnum/Pden;
           
            state[self.N-1] = np.argmax(np.random.multinomial(n=1, pvals=Pi,size=1)) #np.random.choice(self.K, 1, p=Pi) #randsample(1:K,1,true,Pi);

            one_hot = np.squeeze(np.reshape(np.eye(self.K)[state],[1,self.N*self.K] ).astype(np.uint8)).tolist()
            print( state)
            #print one_hot
            for item in one_hot:
                f.write("%s " % item)  
            
            f.write('\n') 
            f.flush(); 
      
            for item in state:
                f2.write("%s " % item)

            f2.write('\n')
            f2.flush();

  
 
        f.close()
        return 
 
    def pXp(self,theta,phi):
         
        return np.array([[ np.cos(theta/2.0)**2, np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(-1j*phi)],\
                         [ np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(1j*phi), np.sin(theta/2.0)**2 ]])

    def mXm(self,theta,phi):

        return np.array([[ np.sin(theta/2.0)**2, -np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(-1j*phi)],\
                         [ -np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(1j*phi), np.cos(theta/2.0)**2 ]])
 
   


#print "POVM", sys.argv[1]
#print "Number_qubits", int(sys.argv[2])
#print "MPS", sys.argv[3]
#print "noise p ", float(sys.argv[4])
#print "Nsamples",int(sys.argv[5])
sampler = PaMPS(POVM=sys.argv[1], Number_qubits = int(sys.argv[2]),MPS=sys.argv[3],p=float(sys.argv[4]))
sampler.samples(Ns=int(sys.argv[5]))
