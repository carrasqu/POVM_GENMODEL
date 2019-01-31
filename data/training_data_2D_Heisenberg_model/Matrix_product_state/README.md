Each Matrix product state is described by two files:

index.txt lists for each of the N tensors in the MPS, its rank, the number of non-zero elements and the dimension of each index:

rank		Number non-zero elements		dim(1)	   …		dim(rank)

tensor.txt contains all the non-zero values of each tensor T (from 1 to N), with their index configuration:

index1		index2		T(index1,index2)
…
… 

index1		index2		index3		T(index1,index2,index3)
…


The order in which the indices appear for each tensor is:
- for the left-most tensor: (right,physical)
- for “bulk" tensors : (right, physical, left)
- for the right-most tensor: (physical, left)


This tensors are obtained using density-matrix renormalization group algorithm 
implemented using ITensor itensor.org/


