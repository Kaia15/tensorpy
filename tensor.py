from typing import Union

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

    def transpose(self,axes) -> 'Tensor':
        pass

    def reshape(self):
        pass

    def flatten(self):
        pass


    
class Test:
    def unittest():
        data = [[[1,2,3]]]
        t = Tensor(data)
        print (t.shape)
        first_zeros = (2,2)
        print (Tensor.zeros(first_zeros).data)
        second_zeros = (2,2,2)
        print (Tensor.zeros(second_zeros).data)
        data = [True, False, 1, 2.0]
        print (Tensor.array(data).data)

Test.unittest()

    