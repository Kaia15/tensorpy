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
            if isinstance(data, int): 
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
            raise ValueError(f"")
        
        if len(shape) == 0:
            return cls([0] * shape[0])
        
        return cls([cls.ones(shape[1:]).data for _ in range(shape[0])])

    @classmethod
    def array(cls,shape: tuple) -> 'Tensor':
        pass

    @classmethod
    def empty(self):
        pass

    def transpose(self,new_shape):
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

Test.unittest()

    