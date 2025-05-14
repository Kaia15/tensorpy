from tensor import Tensor
import numpy as np
import time

def benchmark_transpose():
    data = [[[x + y + z for z in range(200)] for y in range(200)] for x in range(200)]
    t = Tensor(data)
    np_array = np.array(data)
    axes = (1, 2, 0)

    start = time.perf_counter()
    t_transposed = t.transpose(axes)
    end = time.perf_counter()
    print(f"Tensor (sequential) transpose time: {end - start:.4f}s")

    start = time.perf_counter()
    t_mp_transposed = t.multi_processing_transpose(axes)
    end = time.perf_counter()
    print(f"Tensor (multi-processing) transpose time: {end - start:.4f}s")

    start = time.perf_counter()
    np_transposed = np.transpose(np_array, (1, 2, 0))
    end = time.perf_counter()
    print(f"Numpy flatten: {end - start:.4f}s")

def benchmark_c_flatten():
    print ("------------------------------------------------------------")
    data = [[[x + y + z for z in range(100)] for y in range(100)] for x in range(100)]
    t = Tensor(data)
    np_array = np.array(data)
    order = 'C'

    start = time.perf_counter()
    flat_seq = t.flatten(order)
    end = time.perf_counter()
    print(f"Tensor (sequential) C-flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_mp = t.multi_processing_flatten(order)
    end = time.perf_counter()
    print(f"Tensor (multi-processing) C-flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_np = np_array.flatten(order)
    end = time.perf_counter()
    print(f"Numpy C-flatten: {end - start:.4f}s")

def benchmark_f_flatten():
    print ("------------------------------------------------------------")
    data = [[[x + y + z for z in range(100)] for y in range(100)] for x in range(100)]
    t = Tensor(data)
    np_array = np.array(data)
    order = 'F'

    start = time.perf_counter()
    flat_seq = t.flatten(order)
    end = time.perf_counter()
    print(f"Tensor (sequential) F-flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_mp = t.multi_processing_flatten(order)
    end = time.perf_counter()
    print(f"Tensor (multi-processing) F-flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_np = np_array.flatten(order)
    end = time.perf_counter()
    print(f"Numpy F-flatten: {end - start:.4f}s")

def benchmark_arange():
    print ("------------------------------------------------------------")
    # For Tensor class
    # 1 arg: `end`
    start = time.perf_counter()
    t = Tensor.arange(100)
    end = time.perf_counter()
    print(f"Tensor arange(100) time: {end - start:.4f}s")

    # 2 args: `start` & `end`
    start = time.perf_counter()
    t = Tensor.arange(1,100,2)
    end = time.perf_counter()
    print(f"Tensor arange(1, 100, 2) time: {end - start:.4f}s")

    # 3 args: `start`, `end`, and `step`
    start = time.perf_counter()
    t = Tensor.arange(1.0, 100.0, 0.1)
    end = time.perf_counter()
    print(f"Tensor arange(1.0, 100.0, 0.1) time: {end - start:.4f}s")
    
    print ("||")

    # For Numpy
    # 1 arg: `end`
    start = time.perf_counter()
    t = np.arange(100)
    end = time.perf_counter()
    print(f"Numpy arange(100) time: {end - start:.4f}s")

    # 2 args: `start` & `end`
    start = time.perf_counter()
    t = np.arange(1, 100, 2)
    end = time.perf_counter()
    print(f"Numpy arange(1, 100, 2) time: {end - start:.4f}s")

    # 3 args: `start`, `end`, and `step`
    start = time.perf_counter()
    t = np.arange(1.0, 100.0, 0.1)
    end = time.perf_counter()
    print(f"Numpy arange(1, 100, 2) time: {end - start:.4f}s")

def benchmark_reshape():
    # simple test
    # For Tensor class
    print ("------------------------------------------------------------")
    a = [
            [  
                [  
                    [1, 2, 3],     
                    [4, 5, 6]      
                ],
                [  
                    [7, 8, 9],
                    [10, 11, 12]
                ]
            ],
            [  
                [  
                    [13, 14, 15],
                    [16, 17, 18]
                ],
                [  
                    [19, 20, 21],
                    [22, 23, 24]
                ]
            ]
        ]
    start = time.perf_counter()
    t1 = Tensor.reshape(a, (2, 3, 4), order='F')
    end = time.perf_counter()
    print (f"Tensor reshape time: {end - start:.4f}s")
    
    # For Numpy
    start = time.perf_counter()
    t1 = np.reshape(a, (2, 3, 4), order='F')
    end = time.perf_counter()
    print (f"Numpy reshape time: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_transpose()
    benchmark_c_flatten()
    benchmark_f_flatten()
    benchmark_arange()
    benchmark_reshape()