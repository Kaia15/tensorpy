import numpy as np 
data = [
            [
                [x + y + z for z in range(200)]  
                for y in range(200)             
            ]
            for x in range(200)                  
        ]

import time
start = time.perf_counter()
dataT = np.transpose(data,(1,2,0))
end = time.perf_counter()
print ("Progressing time:", end - start)