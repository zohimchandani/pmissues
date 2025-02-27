# mpirun -np 1 --allow-run-as-root python3 test.py --cudaq-full-stack-trace

import cudaq
import cupy as cp
import numpy as np

np.random.seed(0)

cudaq.set_target('nvidia', option='mgpu,fp64')
cudaq.mpi.initialize()

num_qubits = 31
num_ranks = cudaq.mpi.num_ranks()
rank = cudaq.mpi.rank()

print('current rank', rank, 'total ranks', num_ranks)

@cudaq.kernel
def alloc_kernel():
    qubits = cudaq.qvector(num_qubits)

initial_state = cudaq.get_state(alloc_kernel)

@cudaq.kernel
def kernel(state: cudaq.State):
    qubits = cudaq.qvector(state)
    #put gates of interest here

#leave this imports here, do not move it to before kernel definition
from cupy.cuda.memory import MemoryPointer, UnownedMemory

def to_cupy_array(state):
    
    tensor = state.getTensor() #obtain the state that lives on gpu memory and assign it to a variable called tensor
    pDevice = tensor.data() #find location of tensor on gpu memory 
    
    print("Num element =", tensor.get_num_elements()) #prints how many amplitudes live on each rank 
    
    sizeByte = tensor.get_num_elements() * tensor.get_element_size() #calculate memory tensor occupies 
    
    # Use cupy to assign the state memory to a cupy array 
    
    mem = UnownedMemory(pDevice, sizeByte, owner=state)
    memptr = MemoryPointer(mem, 0)
    cupy_array = cp.ndarray(tensor.get_num_elements(),
                              dtype=cp.complex128,
                              memptr=memptr)
    return cupy_array

rank_slice = to_cupy_array(initial_state)


sv = (np.random.randn(2**num_qubits) +  1j * np.random.randn(2**num_qubits)).astype(np.complex128) #initialize 2**n statvector on cpu and allocate each split to gpu 
sv = sv/np.linalg.norm(sv) #normalize statevecor - if you dont normalize, it makes no sense 
split_sv = np.array_split(sv, num_ranks) #split statevec across number of ranks

# Checks
assert len(split_sv) == num_ranks
for sub_state_vec in split_sv:
    assert len(sub_state_vec) == len(rank_slice)

# Mem copy
cp.cuda.runtime.memcpy(rank_slice.data.ptr, split_sv[rank].ctypes.data, split_sv[rank].nbytes, cp.cuda.runtime.memcpyHostToDevice)

result = cudaq.sample(kernel, initial_state)
# result = cudaq.get_state(kernel, initial_state)

# print(result)

assert num_qubits == len(list(result)[0])  

print('done')

cudaq.mpi.finalize()