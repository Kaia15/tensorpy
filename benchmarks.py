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
    data = [[[x + y + z for z in range(100)] for y in range(100)] for x in range(100)]
    t = Tensor(data)
    np_array = np.array(data)
    order = 'C'

    start = time.perf_counter()
    flat_seq = t.flatten(order)
    end = time.perf_counter()
    print(f"Tensor (sequential) flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_mp = t.multi_processing_flatten(order)
    end = time.perf_counter()
    print(f"Tensor (multi-processing) flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_np = np_array.flatten(order)
    end = time.perf_counter()
    print(f"Numpy flatten: {end - start:.4f}s")

def benchmark_f_flatten():
    data = [[[x + y + z for z in range(100)] for y in range(100)] for x in range(100)]
    t = Tensor(data)
    np_array = np.array(data)
    order = 'F'

    start = time.perf_counter()
    flat_seq = t.flatten(order)
    end = time.perf_counter()
    print(f"Tensor (sequential) flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_mp = t.multi_processing_flatten(order)
    end = time.perf_counter()
    print(f"Tensor (multi-processing) flatten time: {end - start:.4f}s")

    start = time.perf_counter()
    flat_np = np_array.flatten(order)
    end = time.perf_counter()
    print(f"Numpy flatten: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_transpose()
    benchmark_c_flatten()
    benchmark_f_flatten()