class Tensor:
    """
    This class is designed & implemented for N-array in Python. 
    It can have basic operations including creation, manipulation, mathematical ops, and poly(s).
    """
    def __init__(self,data):
        self.data = data
        self.shape = self._get_shape(data)

    def _get_shape(self,data):
        shape = []
        print (data)
        while True:
            nxt = len(data)
            shape.append(nxt)
            data = data[0]  # need to check whether it creates a new copy of input data or it points to the original input data 
            if isinstance(data, int): 
                break
        return tuple(shape)
    
class Test:
    def unittest():
        data = [[[1,2,3]]]
        t = Tensor(data)
        print (t.shape)

Test.unittest()

    