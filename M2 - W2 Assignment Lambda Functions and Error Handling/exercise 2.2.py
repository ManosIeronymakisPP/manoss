import time

starting_time = time.time()

from typing import List, Tuple

def sqrt(M:Tuple[str,float])->Tuple[str,float]: #we define a functon that takes a list of numbers 
    K=[] # new list 
    for i in M:
        if type(i)== str: #We modify our code in such a way that when our function read something that it is not a number, returns something meaningfull 
            K.append("this is not a number")
            continue
        if i>=0: #we run the rest of the numbers and find the sqrt of them
            K.append(i**0.5)
        else:
            K.append('we cannot find the sqrt of a negative number')
    return tuple(K) #we turn our list into tuple
            

M= ('we cannot find the sqrt of a negative number', 'we cannot find the sqrt of a negative number', 0.0, 1.0, 1.4142135623730951)

        
print(sqrt(M))


ending_time = time.time()
print(ending_time - starting_time)
        
