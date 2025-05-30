from typing import Union
from itertools import product
from multiprocessing import Pool, cpu_count
import math


class Tensor:
    """
    This class is designed & implemented for N-array in Python. 
    It can have basic operations including creation, manipulation, mathematical ops, and poly(s).
    """

    def __init__(self, data: list) -> 'Tensor':
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
            # need to check whether it creates a new copy of input data or it points to the original input data
            data = data[0]
            if not isinstance(data, list):
                break
        return tuple(shape)

    def _infer_shape(cls, data: list):
        if not isinstance(data, list):
            return ()

        first_shape = cls._infer_shape(data[0])
        for item in data[1:]:
            if cls._infer_shape(item) != first_shape:
                raise ValueError(
                    "Irregular nested list: shape is not consistent.")

        return True

    def _size(data):
        shape = Tensor._get_shape(data)
        return math.prod(list(shape))

    @classmethod
    def ones(cls, shape: tuple) -> 'Tensor':
        """
        TO-DO 
        """
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"Cannot create Tensor with invalid shape {shape}")

        if len(shape) == 1:
            return cls([1] * shape[0])

        return cls([cls.ones(shape[1:]).data for _ in range(shape[0])])

    @classmethod
    def zeros(cls, shape: tuple) -> 'Tensor':
        """
        TO-DO
        """
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"Cannot create Tensor of 1(s) with invalid shape {shape}")

        if len(shape) == 1:
            return cls([0] * shape[0])

        return cls([cls.zeros(shape[1:]).data for _ in range(shape[0])])

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
            strs = [str(arg).split(".")[1]
                    for arg in args if isinstance(arg, float)]
            strs.sort(key=lambda x: len(x))
            max_num_digits = len(strs[-1]) if strs else 0
        data = []

        if is_float:
            start = float(start)
            current = start
            while (current < end and step > 0) or (current > end and step < 0):
                data.append(round(current, max_num_digits))
                current += step
        else:
            data = [j for j in range(start, end, step)]

        return Tensor(data)

    def ndim(self):
        return self.shape

    def _get_value(cor, d):
        for c in cor:
            d = d[c]
        return d

    def _set_value(cor, value, mat):
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
            axes = [i for i in range(ndims - 1, -1, -1)]

        data = self.data

        all_cors = Tensor._all_coords(self.shape)
        formatted_cors = [tuple(x) for x in all_cors]

        coord_value_pairs = [(cor, Tensor._get_value(cor, data))
                             for cor in formatted_cors]

        new_cor_dict = [(tuple(c[i] for i in axes), v)
                        for c, v in coord_value_pairs]

        new_shape = [0] * ndims
        for cor, _ in new_cor_dict:
            for i in range(ndims):
                new_shape[i] = max(new_shape[i], cor[i] + 1)

        initialized_zeros = self.__class__.zeros(new_shape)
        initialized_zeros = initialized_zeros.data

        for cor, val in new_cor_dict:
            Tensor._set_value(cor, val, initialized_zeros)

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
            axes = [i for i in range(ndims - 1, -1, -1)]

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
            raise ValueError(
                f"Cannot reshape the data within invalid {new_shape}")

        if isinstance(new_shape, int):
            return Tensor.flatten(Tensor(a), order) if isinstance(a, list) else Tensor.flatten(a, order)
        else:
            reshaped_data = Tensor.zeros(new_shape).data
            all_cors = Tensor._all_f_coords(
                new_shape) if order == 'F' else Tensor._all_c_coords(new_shape)
            data = Tensor.flatten(Tensor(a), order) if isinstance(
                a, list) else Tensor.flatten(a, order)

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
            raise ValueError(
                "Order must be either C (row-major) or F (column-major)")

        if not order:
            order = 'C'

        shape = self.shape
        data = self.data

        coords = Tensor._all_f_coords(
            shape) if order == 'F' else Tensor._all_c_coords(shape)
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

        coords = Tensor._all_f_coords(
            shape) if order == 'F' else Tensor._all_c_coords(shape)
        input = [(cor, data) for cor in coords]
        flat_data = []

        with Pool(cpu_count()) as pool:
            flat_data = pool.map(Tensor._flatten_get_value, input)

        return flat_data

    def iter_dot(A: Union['Tensor', list, int], B: Union['Tensor', list, int]) -> 'Tensor':
        # A = (d1, d2,..., dk), B = (e1, e2, ..., el-2, el)
        """
        TO-DO
        """

        if any([isinstance(x, int) for x in (A, B)]):
            bothScalar = isinstance(A, int) and isinstance(B, int)
            if bothScalar:
                return A * B

            scalar = A if isinstance(A, int) else B
            mat = A if not isinstance(A, int) else B
            mat = mat.data if isinstance(mat, Tensor) else mat
            m = Tensor._get_shape(mat)

            all_coors = Tensor._all_coords(m)
            init_zeros = Tensor.zeros(m).data
            for c in all_coors:
                Tensor._set_value(c, Tensor._get_value(
                    c, mat) * scalar, init_zeros)

            return Tensor(init_zeros)

        ATensor = isinstance(A, Tensor)
        BTensor = isinstance(B, Tensor)

        if ATensor:
            Adata = A.data
            sA = Tensor._get_shape(Adata)
        else:
            Adata = A
            sA = Tensor._get_shape(A)
        dk = sA[-1]

        if BTensor:
            Bdata = B.data
            sB = Tensor._get_shape(Bdata)
        else:
            Bdata = B
            sB = Tensor._get_shape(B)

        if len(sA) == len(sB) == 1:
            if sA[0] != sB[0]: raise ValueError("")
            return sum([a * b for a,b in zip(Adata, Bdata)])

        el2 = sB[-2]

        # check whether dk matches el-1
        if dk != el2:
            raise ValueError(
                f"The last dimension of first matrix A does not match the second last dimension of second matrix B")

        result_shape = list(sA[:-1]) + list(sB[:-2]) + [sB[-1]]
        dimA = len(sA)
        dimB = len(sB)
        if dimA == dimB == 1:
            total_prod = sum([r * c for r, c in zip(Adata, Bdata)])
            return total_prod

        all_coors = Tensor._fast_all_cords(tuple(result_shape))
        init_zeros = Tensor.zeros(tuple(result_shape)).data

        for c in all_coors:
            prod_sum = 0
            for j in range(dk):
                cA = list(c[:dimA - 1]) + [j]
                cB = list(c[dimA - 1: dimA - 1 + dimB - 2]) + [j] + [c[-1]]
                prod_sum += Tensor._get_value(cA, Adata) * Tensor._get_value(cB, Bdata)
            Tensor._set_value(c, prod_sum, init_zeros)

        return Tensor(init_zeros)

    # def recursive_prod(a: Union[list, 'Tensor'], axis: int = None) -> 'Tensor':
    #     """
    #     Computes the product along a given axis for an n-dimensional array.
    #     Pure Python implementation that handles all cases correctly.
    #     """
    #     if not a:
    #         if axis == 0 or axis == -1:
    #             return 1
    #         raise ValueError(
    #             f"axis {axis} out of bounds for array of dimension 1")

    #     is_tensor = isinstance(a, Tensor)
    #     if is_tensor:
    #         a = a.data

    #     # get the innermost dimension of `a`
    #     if not axis:
    #         flatten_a = Tensor.flatten(Tensor(a), 'C')
    #         return math.prod(flatten_a)

    #     # Helper function for recursive multiplication
    #     def multiply_recursive(a, b):
    #         """Recursively multiply two elements that could be scalars or nested lists"""
    #         if isinstance(a, list) and isinstance(b, list):
    #             if len(a) != len(b):
    #                 raise ValueError(
    #                     "Arrays have incompatible shapes for multiplication")
    #             return [multiply_recursive(a_elem, b_elem) for a_elem, b_elem in zip(a, b)]
    #         elif isinstance(a, list):
    #             return [multiply_recursive(a_elem, b) for a_elem in a]
    #         elif isinstance(b, list):
    #             return [multiply_recursive(a, b_elem) for b_elem in b]
    #         else:
    #             return a * b

    #     # Recursive product computation
    #     def compute_product(arr, current_depth=0):
    #         # If we're not at a list, we have a scalar
    #         if not isinstance(arr, list):
    #             return arr

    #         # If we're at the target axis depth, compute product of this level
    #         if current_depth == axis:
    #             if not arr:
    #                 return 1

    #             # All elements at this level should be processed
    #             result = None
    #             for item in arr:
    #                 item_result = compute_product(item, current_depth + 1)

    #                 if result is None:
    #                     result = item_result
    #                 else:
    #                     # Handle multiplication of different types
    #                     if isinstance(result, list) and isinstance(item_result, list):
    #                         # Element-wise multiplication for lists
    #                         if len(result) != len(item_result):
    #                             raise ValueError(
    #                                 "Arrays have incompatible shapes for multiplication")
    #                         # Recursively multiply corresponding elements
    #                         new_result = []
    #                         for r, i in zip(result, item_result):
    #                             if isinstance(r, list) and isinstance(i, list):
    #                                 # Both are lists, recursively multiply
    #                                 if len(r) != len(i):
    #                                     raise ValueError(
    #                                         "Arrays have incompatible shapes for multiplication")
    #                                 new_result.append(
    #                                     [r_elem * i_elem for r_elem, i_elem in zip(r, i)])
    #                             elif isinstance(r, list):
    #                                 # r is list, i is scalar
    #                                 new_result.append(
    #                                     [r_elem * i for r_elem in r])
    #                             elif isinstance(i, list):
    #                                 # r is scalar, i is list
    #                                 new_result.append(
    #                                     [r * i_elem for i_elem in i])
    #                             else:
    #                                 # Both are scalars
    #                                 new_result.append(r * i)
    #                         result = new_result
    #                     elif isinstance(result, list):
    #                         result = [multiply_recursive(
    #                             r, item_result) for r in result]
    #                     elif isinstance(item_result, list):
    #                         result = [multiply_recursive(
    #                             result, i) for i in item_result]
    #                     else:
    #                         # Both scalars
    #                         result *= item_result

    #             return result if result is not None else 1

    #         if not arr:
    #             return []

    #         return [compute_product(subarr, current_depth + 1) for subarr in arr]

    #     return Tensor(compute_product(a))

    def prod(a: Union[list, 'Tensor'], axis: int = None) -> Union['Tensor', int]:
        """
        TO-DO
        """
        if not a:
            if axis == 0 or axis == -1:
                return 1
            raise ValueError(
                f"axis {axis} out of bounds for array of dimension 1")

        is_tensor = isinstance(a, Tensor)
        if is_tensor:
            a = a.data

        shape = Tensor._get_shape(a)
        # get the innermost dimension of `a`
        if axis == None:
            flatten_a = Tensor.flatten(Tensor(a), 'C')
            return math.prod(flatten_a)

        result_shape = shape[:axis] + shape[axis + 1:]
        init_zeros = Tensor.zeros(result_shape).data
        result_coors = Tensor._fast_all_cords(result_shape)

        for c in result_coors:
            product = 1
            parse_c = list(c)
            for j in range(shape[axis]):
                full_c = parse_c[:axis] + [j] + parse_c[axis:]
                product *= Tensor._get_value(full_c, a)
            # R[i][j][k] = A[i][j][0][k] * A[i][j][1][k] *... * A[i][j][len(shape[axis]) - 1][k]
            Tensor._set_value(c, product, init_zeros)
        return Tensor(init_zeros)

    def sum(a: Union[list, 'Tensor'], axis: int = None) -> Union['Tensor', int]:
        """
        TO-DO
        """
        if not a:
            if axis == 0 or axis == -1:
                return 1
            raise ValueError(
                f"axis {axis} out of bounds for array of dimension 1")

        is_tensor = isinstance(a, Tensor)
        if is_tensor:
            a = a.data

        shape = Tensor._get_shape(a)

        # get the innermost dimension of `a`
        if axis == None:
            flatten_a = Tensor.flatten(Tensor(a), 'C')
            return sum(flatten_a)

        result_shape = shape[:axis] + shape[axis + 1:]
        init_zeros = Tensor.zeros(result_shape).data
        result_coors = Tensor._fast_all_cords(result_shape)

        for c in result_coors:
            total = 0
            parse_c = list(c)
            for j in range(shape[axis]):
                full_c = parse_c[:axis] + [j] + parse_c[axis:]
                total += Tensor._get_value(full_c, a)
            # R[i][j][k] = A[i][j][0][k] * A[i][j][1][k] *... * A[i][j][len(shape[axis]) - 1][k]
            Tensor._set_value(c, total, init_zeros)
        return Tensor(init_zeros)

    # TO-DO: add 'float' type
    def add(A: Union[list, int, 'Tensor'], B: Union[list, int, 'Tensor']) -> Union['Tensor', int]:
        """
        TO-DO
        """
        # Handle scalar cases
        if isinstance(A, int) and isinstance(B, int):
            return A + B

        if isinstance(A, int):
            scalar = A
            mat = B.data if isinstance(B, Tensor) else B
            shape = Tensor._get_shape(mat)
            result = Tensor.zeros(shape)
            for coor in Tensor._all_coords(shape):
                val = Tensor._get_value(coor, mat) + scalar
                Tensor._set_value(coor, val, result.data)
            return result

        if isinstance(B, int):
            return Tensor.add(B, A)

        # Both are tensors/lists
        Adata = A.data if isinstance(A, Tensor) else A
        Bdata = B.data if isinstance(B, Tensor) else B
        sA = Tensor._get_shape(Adata)
        sB = Tensor._get_shape(Bdata)

        # Pad shapes with 1s to make them equal length
        dimA, dimB = len(sA), len(sB)
        if dimA < dimB:
            sA = (1,) * (dimB - dimA) + sA
        elif dimB < dimA:
            sB = (1,) * (dimA - dimB) + sB

        # Determine output shape via broadcasting rules
        result_shape = []
        for a, b in zip(sA, sB):
            if a == b:
                result_shape.append(a)
            elif a == 1:
                result_shape.append(b)
            elif b == 1:
                result_shape.append(a)
            else:
                raise ValueError(f"Shapes {sA} and {sB} are not broadcastable")

        result = Tensor.zeros(result_shape)

        # Perform broadcasted addition
        for coor in Tensor._all_coords(result_shape):
            # Get indices for A and B with broadcasting
            a_index = []
            b_index = []
            for c, a, b in zip(coor, sA, sB):
                a_index.append(0 if a == 1 else c)
                b_index.append(0 if b == 1 else c)

            # Remove padding (1-s) from indices if needed
            if dimA < dimB:
                # (1,..1, d1, d2, ..., dm) -> (d1, d2, ..., dm)
                a_index = a_index[dimB - dimA:]
            elif dimB < dimA:
                # (1,..1, e1, e2, ..., en) -> (e1, e2, ..., en)
                b_index = b_index[dimA - dimB:]

            a_val = Tensor._get_value(a_index, Adata)
            b_val = Tensor._get_value(b_index, Bdata)
            # C[d1][d2]..[dk][ei+1]..[en] = A[d1][d2]..[0 -> dk]..[dm] + B[e1]..[0 -> ei]...[en]
            Tensor._set_value(coor, a_val + b_val, result.data)

        return result

    def lcm(A: Union[list, int, 'Tensor'], B: Union[list, int, 'Tensor']) -> Union['Tensor', int]:
        # Handle scalar cases
        if isinstance(A, int) and isinstance(B, int):
            return math.lcm(A, B)

        if isinstance(A, int):
            scalar = A
            mat = B.data if isinstance(B, Tensor) else B
            shape = Tensor._get_shape(mat)
            result = Tensor.zeros(shape)
            for coor in Tensor._all_coords(shape):
                val = Tensor._get_value(coor, mat) + scalar
                Tensor._set_value(coor, val, result.data)
            return result

        if isinstance(B, int):
            return Tensor.add(B, A)

        # Both are tensors/lists
        Adata = A.data if isinstance(A, Tensor) else A
        Bdata = B.data if isinstance(B, Tensor) else B
        sA = Tensor._get_shape(Adata)
        sB = Tensor._get_shape(Bdata)

        # Pad shapes with 1s to make them equal length
        dimA, dimB = len(sA), len(sB)
        if dimA < dimB:
            sA = (1,) * (dimB - dimA) + sA
        elif dimB < dimA:
            sB = (1,) * (dimA - dimB) + sB

        # Determine output shape via broadcasting rules
        result_shape = []
        for a, b in zip(sA, sB):
            if a == b:
                result_shape.append(a)
            elif a == 1:
                result_shape.append(b)
            elif b == 1:
                result_shape.append(a)
            else:
                raise ValueError(f"Shapes {sA} and {sB} are not broadcastable")

        result = Tensor.zeros(result_shape)

        # Perform broadcasted addition
        for coor in Tensor._all_coords(result_shape):
            # Get indices for A and B with broadcasting
            a_index = []
            b_index = []
            for c, a, b in zip(coor, sA, sB):
                a_index.append(0 if a == 1 else c)
                b_index.append(0 if b == 1 else c)

            # Remove padding (1-s) from indices if needed
            if dimA < dimB:
                # (1,..1, d1, d2, ..., dm) -> (d1, d2, ..., dm)
                a_index = a_index[dimB - dimA:]
            elif dimB < dimA:
                # (1,..1, e1, e2, ..., en) -> (e1, e2, ..., en)
                b_index = b_index[dimA - dimB:]

            a_val = Tensor._get_value(a_index, Adata)
            b_val = Tensor._get_value(b_index, Bdata)
            # C[d1][d2]..[dk][ei+1]..[en] = A[d1][d2]..[0 -> dk]..[dm] + B[e1]..[0 -> ei]...[en]
            Tensor._set_value(coor, math.lcm(a_val, b_val), result.data)

        return result
    
    def __lt__(a: Union[list, float, int, 'Tensor'], thresold):
        is_tensor = isinstance(a, Tensor)
        if not is_tensor:
            if isinstance(a, float) or isinstance(a, int):
                return a < thresold

        sh = a.shape if is_tensor else Tensor._get_shape(a)
        adata = a.data if is_tensor else a
        all_coors = Tensor._all_coords(sh)
        for c in all_coors:
            val = Tensor._get_value(c, adata)
            bval = val < thresold
            Tensor._set_value(c, bval, adata)
        return Tensor(adata)
    
    def __gt__(a, thresold):
        is_tensor = isinstance(a, Tensor)
        if not is_tensor:
            if isinstance(a, float) or isinstance(a, int):
                return a > thresold

        sh = a.shape if is_tensor else Tensor._get_shape(a)
        adata = a.data if is_tensor else a
        all_coors = Tensor._all_coords(sh)
        for c in all_coors:
            val = Tensor._get_value(c, adata)
            bval = val > thresold
            Tensor._set_value(c, bval, adata)
        return Tensor(adata)
    
    def __and__(A, B) -> 'Tensor':
        # Compare A's shape and B's shape
        Atensor = isinstance(A, Tensor)
        Btensor = isinstance(B, Tensor)

        Abool = isinstance(A, bool)
        Bbool = isinstance(B, bool)

        if Abool and Bbool: return (A and B)

        if Abool:
            if not Btensor or isinstance(B, list): raise TypeError("")
            b_sh = Tensor._get_shape(B) if not Btensor else Tensor._get_shape(B.data)
            bdata = B.data if Btensor else B
            all_coors = Tensor._all_coords(b_sh)
            for c in all_coors:
                bval = Tensor._get_value(c, bdata)
                Tensor._set_value(c, (bval and A), bdata)
            return Tensor(bdata)

        if Bbool:
            if not Atensor or isinstance(A, list): raise TypeError("")
            a_sh = Tensor._get_shape(A) if not Atensor else Tensor._get_shape(A.data)
            adata = A.data if Atensor else A
            all_coors = Tensor._all_coords(a_sh)
            for c in all_coors:
                aval = Tensor._get_value(c, adata)
                Tensor._set_value(c, (aval and B), adata)
            return Tensor(adata)
        
        sA = Tensor._get_shape(A) if not Atensor else Tensor._get_shape(A.data)
        adata = A.data if Atensor else A
        sB = Tensor._get_shape(B) if not Btensor else Tensor._get_shape(B.data)
        bdata = B.data if Btensor else B

        if len(sA) == len(sB) == 1:
            if sA[0] != sB[0]: raise ValueError("")
            return Tensor([(a & b) for a,b in zip(adata, bdata)])

        # Pad shapes with 1s to make them equal length
        dimA, dimB = len(sA), len(sB)
        if dimA < dimB:
            sA = (1,) * (dimB - dimA) + sA
        elif dimB < dimA:
            sB = (1,) * (dimA - dimB) + sB

        # Determine output shape via broadcasting rules
        result_shape = []
        for a, b in zip(sA, sB):
            if a == b:
                result_shape.append(a)
            elif a == 1:
                result_shape.append(b)
            elif b == 1:
                result_shape.append(a)
            else:
                raise ValueError(f"Shapes {sA} and {sB} are not broadcastable")

        result = Tensor.zeros(result_shape)

        # Perform broadcasted addition
        for coor in Tensor._all_coords(result_shape):
            # Get indices for A and B with broadcasting
            a_index = []
            b_index = []
            for c, a, b in zip(coor, sA, sB):
                a_index.append(0 if a == 1 else c)
                b_index.append(0 if b == 1 else c)

            # Remove padding (1-s) from indices if needed
            if dimA < dimB:
                # (1,..1, d1, d2, ..., dm) -> (d1, d2, ..., dm)
                a_index = a_index[dimB - dimA:]
            elif dimB < dimA:
                # (1,..1, e1, e2, ..., en) -> (e1, e2, ..., en)
                b_index = b_index[dimA - dimB:]

            a_val = Tensor._get_value(a_index, adata)
            b_val = Tensor._get_value(b_index, bdata)
            # C[d1][d2]..[dk][ei+1]..[en] = A[d1][d2]..[0 -> dk]..[dm] + B[e1]..[0 -> ei]...[en]
            Tensor._set_value(coor, (a_val and b_val), result.data)

        return result

    def positive():
        """
        TO-DO
        """
        pass

    def __neg__(self):
        def negative(a):
            """
            """
            is_tensor = isinstance(a, Tensor)
            if not is_tensor:
                if isinstance(a, float) or isinstance(a, int):
                    return (-1) * a

            sh = a.shape if is_tensor else Tensor._get_shape(a)
            adata = a.data if is_tensor else a
            all_coors = Tensor._all_coords(sh)
            for c in all_coors:
                val = Tensor._get_value(c, adata)
                Tensor._set_value(c, (-1) * val, adata)
            return adata
        return Tensor(negative(self.data))
    
    # DO NOT SUPPORT dtype when sorting a structured array as Numpy does
    def sort(a: Union[list, 'Tensor'], axis: int = None) -> Union['Tensor']:
        is_tensor = isinstance(a, Tensor)

        if is_tensor:
            adata = a.data
        else: 
            adata = a
        
        sh = Tensor._get_shape(adata)

        if axis != None:
            rem_shape = list(sh[:axis]) + list(sh[axis + 1:])
            all_rem_coors = Tensor._all_coords(rem_shape)
            for c in all_rem_coors:
                parsed_c = list(c)
                current = []
                for j in range (sh[axis]):
                    original_c = parsed_c[:axis] + [j] + parsed_c[axis:]
                    val = Tensor._get_value(original_c, adata)
                    current += [val]
                current.sort()
                for j in range (sh[axis]):
                    original_c = parsed_c[:axis] + [j] + parsed_c[axis:]
                    Tensor._set_value(original_c, current[j], adata)
            
            return Tensor(adata)
        else:
            flatten_a = Tensor.flatten(Tensor(adata), 'C')
            flatten_a.sort()
            return Tensor(flatten_a)

class Test:
    @staticmethod
    def unittest():
        test_array = [
            [
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                    [[13, 14, 15], [16, 17, 18]]
                ],
                [
                    [[19, 20, 21], [22, 23, 24]],
                    [[25, 26, 27], [28, 29, 30]],
                    [[31, 32, 33], [34, 35, 36]]
                ]
            ]
        ]

        print(Tensor.prod(test_array, axis=2).data)

        A = [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],  # 1st 3x4 block
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]  # 2nd 3x4 block
        ]

        B = [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20]],  # 1st 4x5 block
            [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30], [
                31, 32, 33, 34, 35], [36, 37, 38, 39, 40]]  # 2nd 4x5 block
        ]

        print(Tensor.iter_dot(A, B).data)

        A1 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        B1 = [[1, 2], [3, 4], [5, 6]]
        B1 = 2

        print(Tensor.iter_dot(A1, B1).data)

        A2 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        print(Tensor.sum(A2, axis=2).data)

        A = [[1], [2], [3]]   # Shape (3, 1)
        B = [[4, 5, 6]]       # Shape (1, 3)
        # Output: [[5, 6, 7], [6, 7, 8], [7, 8, 9]]
        print(Tensor.add(A, B).data)

        A = [[[1, 2, 3]]]     # Shape (1, 1, 3)
        B = [[[4], [5]]]      # Shape (1, 2, 1)
        print(Tensor.add(A, B).data)

        A = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]  # Shape (2, 2, 2)

        B = [
            [[10], [20]]
        ]  # Shape (1, 2, 1)
        print(Tensor.add(A, B).data)

        # B = [
        #     [100, 200, 300, 400],  # Row 0
        #     [500, 600, 700, 800]   # Row 1
        # ]
        # Raise ValueError
        # print (Tensor.add(A, B))

        A = 4
        B = 6
        print(Tensor.lcm(A, B))

        A = [
            [  # Block 0
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ],
            [  # Block 1
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ]
        ]

        A5 = Tensor(A)
        A6 = -A5
        print(A6.data)

        A = 4
        A7 = -A
        print(A7)

        A = [[1,2,3,4]]
        B = [1,2,3,4]
        C = Tensor(A) > 1
        D = Tensor(B) < 3
        print (C.data)
        print (D.data)
        E = C & D 
        print (E.data)

        A = 4 
        B = 5
        C = A < 2
        D = B > 3
        print (C & D)

        A = [[1,4],[3,1]]
        print(Tensor.sort(A).data)
        print (Tensor.sort(A, axis = 0).data)

"""
Why do we need to test multiprocessing in main() stack?
"""
if __name__ == "__main__":
    Test.unittest()
