#from __future__ import print_function
import numpy as np

def readMPS(MPSf='tensor.txt', N=16,convert=True):
 
    indices= []; 
    with open("index.txt") as f:
        for line in f:
            int_list = [int(x) for x in line.split()]
            indices.append(int_list)
   
    MPS = []
    # initialization of MPS
    MPS.append(np.zeros((indices[0][2],indices[0][3])))
    for i in range(1,N-1):
        MPS.append(np.zeros((indices[i][2],indices[i][3],indices[i][4])))

    MPS.append(np.zeros((indices[N-1][2],indices[N-1][3])))

       
    arrays = [np.array(map(float, line.split())) for line in open(MPSf)]
    #print(arrays) 
    # 1st matrix
    counter=0
    #print indices 
    for line in range(indices[0][1]):
        #print( MPS[0].shape)
        #print( arrays[counter][0],arrays[counter][1],arrays[counter][2]  )
        MPS[0][ int(arrays[counter][0])-1,int(arrays[counter][1])-1]= arrays[counter][2] 
        counter+=1
 
    counter+=1 # skip hole in the file (Giacomo's format)

    # populating the MPS in the "bullk"
    for i in range(1,N-1):
        for line in range(indices[i][1]):
            MPS[i][ int(arrays[counter][0])-1,int(arrays[counter][1])-1,int(arrays[counter][2])-1]= arrays[counter][3] 
            counter+=1
        counter+=1 # skip hole in the file (Giacomo's format)

    # reading last matrix 
    #print indices
    for line in range(indices[N-1][1]):
        #print( MPS[N-1].shape)
        #print( arrays[counter][0],arrays[counter][1],arrays[counter][2])
        MPS[N-1][ int(arrays[counter][0])-1,int(arrays[counter][1])-1]= arrays[counter][2]
        counter+=1

     
    # Permute to my format from Giacomos ordering 
    if convert == True:
        MPS[0] = np.transpose(MPS[0],(1,0));
        for i in range(1,N-1):
          MPS[i] = np.transpose(MPS[i],[2,1,0]);

     
      #% MPS{N} = permute(MPS{N},[2,1]);


 
    return MPS

