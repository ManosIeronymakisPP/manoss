from typing import List, Dict

def dup(L:List[int])->List[int]:
    return list(dict.fromkeys(L)) # we create a dictionary that has the elements of our list as keys amd after we convert the dictionary into a list.


L:List[int]= [1,2,3,2,3,1]
print(dup(L))
