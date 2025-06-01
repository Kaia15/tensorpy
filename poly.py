import math
from typing import Union
from collections import deque, defaultdict

class Polynomial:
    def __init__(self, c, r: bool = False):
        self.c = c
        self.r = r
        if self.r:
            total_p = Polynomial([1, self.c[0] * (-1)])
            for i in range (1, len(self.c)):
                total_p = Polynomial.multiply(total_p.c, [1, self.c[i] * (-1)])
            self.c = total_p.c 
            self.r = False

    def __call__(self, x: Union[int, float]):
        if not self.r: 
            return sum([self.c[i] * (x**(len(self.c) - 1 - i)) for i in range (len(self.c))])
        return math.prod([(x - self.c[i]) for i in range (len(self.c))])

    def __str__(self):
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
    
    def add(c1: list[Union[int, float]], c2: list[Union[int, float]]):

        m,n = len(c1), len(c2)
        min_coeffs = min(m, n)
        result = deque()

        for j in range (-1, (-1) * min_coeffs - 1, -1):
            result.appendleft(c1[j] + c2[j])
            m -= 1
            n -= 1
        
        if m >= 1: 
            for j in range (m - 1, -1, -1):
                result.appendleft(c1[j])
        
        if n >= 1:
            for j in range (n - 1, -1, -1):
                result.appendleft(c2[j])

        return Polynomial(list(result))

    def multiply(c1: list[Union[int, float]], c2: list[Union[int, float]]):
        m,n = len(c1), len(c2)
        m1,n1 = m - 1, n - 1
        result = []

        for i in range(m):
            for j in range (n):
                result.append((c1[i] * c2[j], (m1 - i) + (n1 - j)))

        
        final = {}
        max_id = 0
        for x,y in result:
            if y not in final:
                final[y] = 0
            final[y] += x
            max_id = max(max_id, y)
        
        c = [final[max_id - i] for i in range (max_id + 1) if (max_id - i) in final]

        return Polynomial(c)
    
    def square(self):
        return Polynomial.multiply(self.c, self.c)
    
    def __add__(p1, p2):
        if not isinstance(p1, Polynomial) or not isinstance(p2, Polynomial):
            raise TypeError(f"")
        c1, c2 = p1.c, p2.c

        return Polynomial.add(c1, c2)
    
    def __mul__(p1, p2):
        if not isinstance(p1, Polynomial) or not isinstance(p2, Polynomial):
            raise TypeError(f"")
        
        c1, c2 = p1.c, p2.c 
        return Polynomial.multiply(c1, c2)
    
    def __getitem__(self, exp: int):
        if exp < 0 or exp > len(self.c):
            raise ValueError(f"")
        
        return self.c[-(exp + 1)]
    
    def __pow__(self, other: int):
        if other < 0: raise ValueError(f"")
        if other == 1:
            return self
        if other % 2 == 1: return Polynomial.multiply(self.__pow__(other - 1).c, self.c)
        else: return Polynomial.multiply(self.__pow__(other // 2).c, self.__pow__(other // 2).c) 
    
    def get_coeffs(self):
        return self.c
    
    def deriv(self):
        pass 

    def integ(self):
        pass
        
        