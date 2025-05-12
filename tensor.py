from typing import Union
from collections import defaultdict
from itertools import product
from multiprocessing import Pool, cpu_count
import time 

class Tensor:
    """
    This class is designed & implemented for N-array in Python. 
    It can have basic operations including creation, manipulation, mathematical ops, and poly(s).
    """
    def __init__(self,data: list) -> 'Tensor':
        """
        TO-DO
        """
        self.data = data
        self.shape = self._get_shape(data)

    def _get_shape(self,data: list) -> tuple:
        """
        TO-DO
        """
        shape = []
        while True:
            nxt = len(data)
            shape.append(nxt)
            data = data[0]  # need to check whether it creates a new copy of input data or it points to the original input data 
            if not isinstance(data, list): 
                break
        return tuple(shape)

    @classmethod
    def ones(cls,shape: tuple) -> 'Tensor':
        """
        TO-DO 
        """
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"Cannot create Tensor with invalid shape {shape}")
        
        if len(shape) == 1:
            return cls([1] * shape[0])
        
        return cls([cls.ones(shape[1:]).data for _ in range(shape[0])])
    
    @classmethod
    def zeros(cls,shape: tuple) -> 'Tensor':
        """
        TO-DO
        """
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"Cannot create Tensor of 1(s) with invalid shape {shape}")
        
        if len(shape) == 1:
            return cls([0] * shape[0])
        
        return cls([cls.zeros(shape[1:]).data for _ in range(shape[0])])

    @classmethod
    def array(cls, data: Union[list, tuple], ndmin: int = 0) -> 'Tensor':
        """
        TO-DO
        notes about `map()`
        """
        if any(isinstance(x, str) for x in data): 
            raise TypeError(f"Our package only processes int, float, and bool types.")
        
        # instead of using "Tensor(data)", we just need to call "cls(data)"
        # upcast if there exists float in the input data
        exist_float = any(isinstance(x, float) for x in data)
        exist_bool = any(isinstance(x, bool) for x in data)

        def bool2int(x):
            if isinstance(x, bool): return 1 if True else 0
            return x
        if exist_bool:
            data = map(bool2int, data)

        if exist_float:
            data = map(float, data)

        while ndmin > 0:
            data = [data]
            ndmin -= 1

        return cls(list(data))


    @classmethod
    def empty(self):
        pass

    def simple_transpose(self, axes: tuple = None):
        """
        TO-DO
        """
        start = time.perf_counter()
        ndims = len(self.shape)
        if not axes:
            axes = [i for i in range (ndims - 1,-1,-1)]
        
        data = self.data
        
        def all_coords(shape):
            if not shape:
                return [[]]  
            result = []
            for i in range(shape[0]):
                for tail in all_coords(shape[1:]):
                    result.append([i] + tail)
            return result

        all_cors = all_coords(self.shape)
        formatted_cors = [tuple(x) for x in all_cors]
        
        def get_value(cor,d):
            for c in cor:
                d = d[c]
            return d
        
        coord_value_pairs = [(cor, get_value(cor, data)) for cor in formatted_cors]

        
        new_cor_dict = [(tuple(c[i] for i in axes), v) for c, v in coord_value_pairs]

        new_shape = [0] * ndims
        for cor, _ in new_cor_dict:
            for i in range (ndims):
                new_shape[i] = max(new_shape[i], cor[i] + 1)

        initialized_zeros = self.__class__.zeros(new_shape)
        initialized_zeros = initialized_zeros.data
        

        def set_value(cor,value,mat):
            for c in cor[:-1]:
                mat = mat[c]
            mat[cor[-1]] = value
        for cor,val in new_cor_dict:
            set_value(cor,val,initialized_zeros)

        end = time.perf_counter()
        print("Progressing time:", end - start)
        return Tensor(initialized_zeros)
    
    @staticmethod
    def _worker(args):
        cor, value, axes = args
        new_cor = tuple(cor[i] for i in axes)
        return new_cor, value
    
    def fast_transpose(self, axes: tuple = None):
        """
        TO-DO
        """
        start = time.perf_counter()
        ndims = len(self.shape)
        if not axes:
            axes = [i for i in range (ndims - 1,-1,-1)]

        shape = self.shape
        data = self.data
        
        new_shape = [shape[ax] for ax in axes]

        def all_cords(shape):
            return list(product(*[range(dim) for dim in shape]))

        def get_value(cor, d):
            for c in cor:
                d = d[c]
            return d

        cors = all_cords(shape)
        args = [(cor, get_value(cor, data), axes) for cor in cors]

        with Pool(cpu_count()) as pool:
            mapped = pool.map(Tensor._worker, args)

        initialized_zeros = self.__class__.zeros(tuple(new_shape)).data

        def set_value(cor, value, mat):
            for c in cor[:-1]:
                mat = mat[c]
            mat[cor[-1]] = value

        for new_cor, value in mapped:
            set_value(new_cor, value, initialized_zeros)

        end = time.perf_counter()
        print("Progressing time:", end - start)
        return Tensor(initialized_zeros)


    def reshape(self):
        pass

    def flatten(self):
        pass


    
class Test:
    @staticmethod
    def unittest():
        data = [
            [
                [x + y + z for z in range(200)]  
                for y in range(200)             
            ]
            for x in range(200)                  
        ]
        t1 = Tensor(data)
        t2 = Tensor(data)
        simple_transposed = t1.simple_transpose(axes=(1,2,0))
        fast_transposed = t2.fast_transpose(axes=(1,2,0))
        # print (t.data)
        # print (simple_transposed.data)
        # print (fast_transposed.data)


"""
Why do we need to test multiprocessing in main() stack?
"""
if __name__ == "__main__":
    Test.unittest()


    