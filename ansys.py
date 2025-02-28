# mpirun -np xx --allow-run-as-root python3 test.py --cudaq-full-stack-trace

import cudaq
import cupy as cp
import numpy as np

np.random.seed(0)

cudaq.set_target('nvidia', option='mgpu,fp64')
cudaq.mpi.initialize()

num_qubits = 30
num_ranks = cudaq.mpi.num_ranks()
rank = cudaq.mpi.rank()

print('current rank', rank, 'total ranks', num_ranks)


@cudaq.kernel
def alloc_kernel():
    qubits = cudaq.qvector(num_qubits)

#allocate a n qubit statevec on gpu memory taking advanatge of mgpu functionality 
alloc_state = cudaq.get_state(alloc_kernel)

@cudaq.kernel
def kernel(state: cudaq.State):
    qubits = cudaq.qvector(state)
    #put gates of interest here

#leave these imports here, do not move it to before kernel definition
from cupy.cuda.memory import MemoryPointer, UnownedMemory

def to_cupy_array(allocated_state):
    
    #obtain the allocated state that lives on gpu memory and assign it to a variable
    state_on_gpu = allocated_state.getTensor()
    
    #find memory address of the allocated state on gpu memory  
    mem_address_state_on_gpu = state_on_gpu.data() 
    
    #prints how many amplitudes live on each rank. Sum of amp per rank = 2**nqubits
    print("Number of amplitudes on each rank =", state_on_gpu.get_num_elements()) 
    
    #calculate memory the state occupies in bytes per rank
    sizeByte = state_on_gpu.get_num_elements() * state_on_gpu.get_element_size()
    print("size of allocated state distributed on each rank =", sizeByte) 

    #use cupy to assign the statevec memory to a cupy array 
    mem = UnownedMemory(mem_address_state_on_gpu, sizeByte, owner=allocated_state)
    
    memptr = MemoryPointer(mem, 0)
    
    cupy_array = cp.ndarray(state_on_gpu.get_num_elements(),
                              dtype=cp.complex128,
                              memptr=memptr)
    return cupy_array

#assign initial allocated distributed state to a cupy array on each rank 
rank_slice = to_cupy_array(alloc_state)

#initialize 2**n statvector on cpu and split it 
sv = (np.random.randn(2**num_qubits) +  1j * np.random.randn(2**num_qubits)).astype(np.complex128) 

sv = sv/np.linalg.norm(sv) #normalize statevecor - if you dont normalize, results make no sense
split_sv = np.array_split(sv, num_ranks) #split statevec 

#num of splits == num of ranks 
#num of elements in a split == num of elemments blocked on gpu memory waiting to be populated 
assert len(split_sv) == num_ranks 
for sub_state_vec in split_sv: 
    assert len(sub_state_vec) == len(rank_slice)

#copy split statevecs to the cupy arrays holding the initial allocated state  
cp.cuda.runtime.memcpy(rank_slice.data.ptr, split_sv[rank].ctypes.data, split_sv[rank].nbytes, cp.cuda.runtime.memcpyHostToDevice)

#execute the kernel with the initial allocated state which has now been populated with the state of interest
result = cudaq.sample(kernel, alloc_state)
# result = cudaq.get_state(kernel, alloc_state)

# print(result)

assert num_qubits == len(list(result)[0])  

print('done')

cudaq.mpi.finalize()