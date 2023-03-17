import time

starting_time = time.time() #The time() function from the time module gets the current time.
                            #We can use time() to get the starting time,
                            #the ending time and then take the time difference to get the total time that our program is executing

from typing import List, Tuple

def sqrt(L:List[int])->Tuple[str,float]: #we define a functon that takes a list of numbers 
    K=[] # new list 
    for i in L: # we run all the items of the list L and return the numbers into the list K, depending if our number is positive or negative
        if i>=0:
            K.append(i**0.5)
        else:
            K.append('we cannot find the sqrt of a negative number')
    return tuple(K) #we turn our list into tuple
            
            
L=[-2,-1,0,1,2]
print(sqrt(L))

ending_time = time.time()
print(ending_time - starting_time) #estimate time 0.010006427764892578
        

