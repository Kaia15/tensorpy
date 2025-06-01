import math
from typing import Union

class Polynomial:
    def __init__(self, c, r: bool = False):
        self.c = c
        self.r = r

    def __call__(self, x: Union[int, float]):
        if not self.r: 
            return sum([self.c[i] * (x**(len(self.c) - 1 - i)) for i in range (len(self.c))])
        return math.prod([(x - self.c[i]) for i in range (len(self.c))])

    def __str__(self):
        if self.r:
            return " * ".join([f"(x - {self.c[i]})" for i in range(len(self.c))])
        poly_st = []
        for i in range (len(self.c)):
            if self.c[i] == 0: continue
            if i == len(self.c) - 2:
                poly_st += [f"{self.c[i]}x"]
            elif i == len(self.c) - 1:
                poly_st += [str(self.c[i])]
            else:
                poly_st += [f"{self.c[i]}x^{len(self.c) - 1 - i}"] if self.c[i] != 1 else [f"x^{len(self.c) - 1 - i}"]
                
        return " + ".join(poly_st)