import time

starting_time = time.time()

from typing import List, Tuple, Callable

lambda_function_sqrt:Callable[[List[int]],Tuple[str,float]]= lambda x: x**0.5 if x>=0 else "we cannot find the sqrt of a negative number"
print(tuple([lambda_function_sqrt(el) for el in [-2,-1,0,1,2]]))

ending_time = time.time()
print(ending_time - starting_time) #estimate time 0.00900721549987793
