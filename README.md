# Issues scaling mgpu code on perlmutter 

Follow instructions from [this]([url](https://github.com/zohimchandani/cudaq-perlmutter/tree/main)) repos README to get the right CUDA-Q Perlmutter settings 

```
Num_qubits = 30, N = 1, n=4, 36322883.out - successful 
Num_qubits = 30, N = 2, n=8, 36323112.out - successful  

Num_qubits = 31, N = 1, n=4, 36325542.out - error 
Num_qubits = 31, N = 2, n=8, 36323415.out - error  
```
