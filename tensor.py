from typing import Union
from collections import defaultdict
from itertools import product
from multiprocessing import Pool, cpu_count
import numpy as np
import math

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
        self.shape = Tensor._get_shape(data)

    def _get_shape(data: list) -> tuple:
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
    
    def _infer_shape(cls, data: list):
        if not isinstance(data, list):
            return ()
        
        first_shape = cls._infer_shape(data[0])
        for item in data[1:]:
            if cls._infer_shape(item) != first_shape:
                raise ValueError("Irregular nested list: shape is not consistent.")
        
        return True
    
    def _size(data):
        shape = Tensor._get_shape(data)
        return math.prod(list(shape))

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

    # @classmethod
    # def array(cls, data: Union[list, tuple], ndmin: int = 0) -> 'Tensor':
    #     """
    #     TO-DO
    #     notes about `map()`
    #     """
    #     if any(isinstance(x, str) for x in data): 
    #         raise TypeError(f"Our package only processes int, float, and bool types.")
        
    #     # instead of using "Tensor(data)", we just need to call "cls(data)"
    #     # upcast if there exists float in the input data
    #     same_type = True
    #     # check whether the whole list is same-typed
    #     exist_float = any(isinstance(x, float) for x in data)
    #     exist_bool = any(isinstance(x, bool) for x in data)

    #     def bool2int(x):
    #         if isinstance(x, bool): return 1 if True else 0
    #         return x
    #     if exist_bool:
    #         data = map(bool2int, data)

    #     if exist_float:
    #         data = map(float, data)

    #     while ndmin > 0:
    #         data = [data]
    #         ndmin -= 1

    #     return cls(list(data))


    @classmethod
    def empty(self):
        pass

    
    def arange(*args) -> 'Tensor':
        """
        1. If any argument is float -> do float addition 
        2. No floats -> use .range() in Python 
        """
        num_args = len(args)
        if num_args == 1: 
            start, end = 0, args[0]
            step = 1
        elif num_args == 2:
            start, end = args
            step = 1
        else:
            start, end, step = args
        
        is_float = any([isinstance(arg, float) for arg in args])
        if is_float:
            strs = [str(arg).split(".")[1] for arg in args if isinstance(arg,float)]
            strs.sort(key = lambda x: len(x))
            max_num_digits = len(strs[-1]) if strs else 0
        data = []
        
        if is_float:
            start = float(start)
            current = start
            while (current < end and step > 0) or (current > end and step < 0):
                data.append(round(current,max_num_digits))
                current += step
        else:
            data = [j for j in range(start, end, step)]
        # print (data)

        return Tensor(data)

    def ndim(self):
        return self.shape

    def _get_value(cor,d):
        for c in cor:
            d = d[c]
        return d
    
    def _set_value(cor,value,mat):
            for c in cor[:-1]:
                mat = mat[c]
            mat[cor[-1]] = value

    def _all_coords(shape):
        if not shape:
            return [[]]  
        result = []
        for i in range(shape[0]):
            for tail in Tensor._all_coords(shape[1:]):
                result.append([i] + tail)
        return result
    
    def _fast_all_cords(shape):
        return list(product(*[range(dim) for dim in shape]))

    def transpose(self, axes: tuple = None) -> 'Tensor':
        """
        TO-DO
        """
        ndims = len(self.shape)
        if not axes:
            axes = [i for i in range (ndims - 1,-1,-1)]
        
        data = self.data    

        all_cors = Tensor._all_coords(self.shape)
        formatted_cors = [tuple(x) for x in all_cors]
 
        coord_value_pairs = [(cor, Tensor._get_value(cor, data)) for cor in formatted_cors]
   
        new_cor_dict = [(tuple(c[i] for i in axes), v) for c, v in coord_value_pairs]

        new_shape = [0] * ndims
        for cor, _ in new_cor_dict:
            for i in range (ndims):
                new_shape[i] = max(new_shape[i], cor[i] + 1)

        initialized_zeros = self.__class__.zeros(new_shape)
        initialized_zeros = initialized_zeros.data
                
        for cor,val in new_cor_dict:
            Tensor._set_value(cor,val,initialized_zeros)

        return Tensor(initialized_zeros)
    
    @staticmethod
    def _transpose_worker(args):
        cor, value, axes = args
        new_cor = tuple(cor[i] for i in axes)
        return new_cor, value
    
    def multi_processing_transpose(self, axes: tuple = None) -> 'Tensor':
        """
        TO-DO
        """
        ndims = len(self.shape)
        if not axes:
            axes = [i for i in range (ndims - 1,-1,-1)]

        shape = self.shape
        data = self.data
        
        new_shape = [shape[ax] for ax in axes]

        cors = Tensor._fast_all_cords(shape)
        args = [(cor, Tensor._get_value(cor, data), axes) for cor in cors]

        with Pool(cpu_count()) as pool:
            mapped = pool.map(Tensor._transpose_worker, args)

        initialized_zeros = self.__class__.zeros(tuple(new_shape)).data

        for new_cor, value in mapped:
            Tensor._set_value(new_cor, value, initialized_zeros)

        return Tensor(initialized_zeros)

    def reshape(a: Union['Tensor', list], new_shape: Union[tuple, int], order: str = None) -> 'Tensor':
        if not order: 
            order = 'C'
        
        # check to call infer_shape (for inconsistent list)
        if not isinstance(a, Tensor):
            a = Tensor(a)

        num_elements = Tensor._size(a.data)
        actual_num_elements = math.prod(list(new_shape))
        if actual_num_elements != num_elements:
            raise ValueError(f"Cannot reshape the data within invalid {new_shape}")
            
        if isinstance(new_shape, int):
            return Tensor.flatten(Tensor(a), order) if isinstance(a, list) else Tensor.flatten(a, order)
        else:
            reshaped_data = Tensor.zeros(new_shape).data
            all_cors = Tensor._all_f_coords(new_shape) if order == 'F' else Tensor._all_c_coords(new_shape)
            data = Tensor.flatten(Tensor(a), order) if isinstance(a, list) else Tensor.flatten(a, order)
        
            for new_cor, value in zip(all_cors, data):
                Tensor._set_value(new_cor, value, reshaped_data)

            return Tensor(reshaped_data)

    @classmethod
    def _all_f_coords(cls, shape):
        if not shape:
            return [[]]
        result = []
        for tail in cls._all_f_coords(shape[1:]):
            for i in range(shape[0]):
                result.append([i] + tail)
        return result
    
    @classmethod
    def _all_c_coords(cls, shape):
        if not shape:
            return [[]]
        result = []
        for i in range(shape[0]):
            for tail in cls._all_c_coords(shape[1:]):
                result.append([i] + tail)
        return result

    def flatten(self, order: str = None) -> list:
        """
        @param: order ('C' or 'F')
        """
        if order not in ['C', 'F']:
            raise ValueError("Order must be either C (row-major) or F (column-major)")
        
        if not order: 
            order = 'C'

        shape = self.shape 
        data = self.data 

        coords = Tensor._all_f_coords(shape) if order == 'F' else Tensor._all_c_coords(shape)
        flat_data = [Tensor._get_value(cor, data) for cor in coords]

        return flat_data

    @staticmethod
    def _flatten_get_value(args):
        cor, d = args
        for c in cor:
            d = d[c]
        return d

    def multi_processing_flatten(self, order: str = None) -> list:
        """
        @param: order ('C' or 'F')
        """
        if not order: 
            order = 'C'
        shape = self.shape 
        data = self.data 

        coords = Tensor._all_f_coords(shape) if order == 'F' else Tensor._all_c_coords(shape)
        input = [(cor, data) for cor in coords]
        flat_data = []

        with Pool(cpu_count()) as pool:
            flat_data = pool.map(Tensor._flatten_get_value, input)

        return flat_data

    def dot(x1: Union['Tensor', list, int], x2: Union['Tensor', list, int]) -> 'Tensor':

        if any([isinstance(x, int) for x in (x1, x2)]): 
            bothScalar = isinstance(x1, int) and isinstance(x2, int)
            if bothScalar: return x1 * x2

            scalar = x1 if isinstance(x1, int) else x2
            mat = x1 if not isinstance(x1, int) else x2 
            mat = mat.data if isinstance(mat, Tensor) else mat 
            m = Tensor._get_shape(mat)

            all_coors = Tensor._all_coords(m)
            prod_data = [Tensor._get_value(c, mat) * scalar for c in all_coors]
            return Tensor(prod_data)

        if isinstance(x1, list):
            x1_shape = Tensor._get_shape(x1)
        else:
            x1_shape = Tensor._get_shape(x1.data)

        if isinstance(x2, list):
            x2_shape = Tensor._get_shape(x2)
        else:
            x2_shape = Tensor._get_shape(x2.data)

        # print (x1_shape, x2_shape)

        if len(x1_shape) == len(x2_shape) == 1:
            x1_data = x1.data if isinstance(x1, Tensor) else x1
            x2_data = x2.data if isinstance(x2, Tensor) else x2
            total_prod = sum([r * c for r,c in zip(x1_data, x2_data)])
            return total_prod

        if x1_shape[-1] != x2_shape[-2]: 
            raise ValueError(f"Incompatible shapes for dot product")
        
        m = x1_shape[-1]
        
        # get the last dimension of x1
        all_coors = Tensor._all_coords(x1_shape)
        x1_mat = x1.data if isinstance(x1, Tensor) else x1
        x1_data = [Tensor._get_value(c, x1_mat) for c in all_coors]
        # print (x1_data)
        rows = []
        for i in range (0, len(x1_data), m):
            row = x1_data[i : i + m]
            rows.append(row)
        # print (rows)
        
        all_f_coors = Tensor._all_f_coords(x2_shape)
        x2_mat = x2.data if isinstance(x2, Tensor) else x2
        x2_data = [Tensor._get_value(c, x2_mat) for c in all_f_coors]
        cols = []
        for j in range (0, len(x2_data), m):
            col = x2_data[j : j + m]
            cols.append(col)
        # print (cols)
        C_order = []
        for row in rows:
            for col in cols:
                dot_val = sum(r * c for r, c in zip(row, col))
                C_order.append(dot_val)
        # print (C_order)
        # TO-DO: distribute all elements in C_order list into a matrix with final shape 
        final_shape = tuple(list(x1_shape[:-1]) + list(x2_shape[-1:]))
        final_data = Tensor.zeros(final_shape).data
        all_coors = Tensor._all_coords(final_shape)
        for c, value in zip(all_coors, C_order):
            Tensor._set_value(c, value, final_data)
        
        return Tensor(final_data)
    
    def add(x1, x2):
        pass
    
class Test:
    @staticmethod
    def unittest():
       A = [
        [  # Batch 0
            [[ 1, 2, 3, 4, 5, 6],
            [ 7, 8, 9,10,11,12],
            [13,14,15,16,17,18],
            [19,20,21,22,23,24]],

            [[25,26,27,28,29,30],
            [31,32,33,34,35,36],
            [37,38,39,40,41,42],
            [43,44,45,46,47,48]],
        ],

        [  # Batch 1
            [[49,50,51,52,53,54],
            [55,56,57,58,59,60],
            [61,62,63,64,65,66],
            [67,68,69,70,71,72]],

            [[73,74,75,76,77,78],
            [79,80,81,82,83,84],
            [85,86,87,88,89,90],
            [91,92,93,94,95,96]],
        ],

        [  # Batch 2
            [[97, 98, 99,100,101,102],
            [103,104,105,106,107,108],
            [109,110,111,112,113,114],
            [115,116,117,118,119,120]],

            [[121,122,123,124,125,126],
            [127,128,129,130,131,132],
            [133,134,135,136,137,138],
            [139,140,141,142,143,144]],
        ]
        ]

       B = [
        [  # Only 1 batch (to be broadcast)
            [1, 2, 3, 4, 5, 6, 7],
            [8, 9,10,11,12,13,14],
            [15,16,17,18,19,20,21],
            [22,23,24,25,26,27,28],
            [29,30,31,32,33,34,35],
            [36,37,38,39,40,41,42],
        ]
        ]

       Tensor.dot(A, B)
        
"""
Why do we need to test multiprocessing in main() stack?
"""
if __name__ == "__main__":
    Test.unittest()


    