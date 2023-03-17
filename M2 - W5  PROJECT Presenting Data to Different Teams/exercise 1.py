import time
from typing import Callable

'''This program defines a custom decorator "measure_time" that can be used to measure the execution time of a function.
   The measure_time decorator takes a single argument, which is the function that the decorator will be applied to.
   Inside the decorator, a new function "execution_time" is defined.
   This function will be used to measure the execution time of the original function.'''

def measure_time(func: Callable) -> Callable:
    
    '''The "execution_time" takes arguments (*args, **kwargs) that allows "the measure_time" to accept any number of arguments.
    Inside the "execution_time", the current time is recorded in the start variable using time.time().'''
    
    def execution_time(*args: tuple, **kwargs: dict) -> any:
        start = time.time()
        
        '''The original function is then executed by calling func(*args, **kwargs) and the result of the function is assigned to the result .
        The current time is recorded in the end variable.
        The execution time of the function is calculated by subtracting the start time from the end time and is printed in the format "'Executed {func.__name__} in {end - start} seconds.' ."
        The function returns the result.'''
        
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Executed {func.__name__} in {end - start} seconds.')
        return result

    '''Finally, the "execution_time" function is returned by the decorator.'''
    
    return execution_time

'''The "my_function" is decorated with the @measure_time decorator, which means that the "measure_time" decorator will be applied to the "my_function".
   The my_function sleeps for 2 seconds, when the function is called, it will execute and the execution time will be printed.
   The value of 2 for time.sleep(2) is arbitrary and can be adjusted to any value as per the requirement.
   It is not necessary to set time.sleep(2) and not something else, it could be any value that we think that function take a significant amount of time to execute.
   The idea is to set a value that would reflect the time that the real function needs to execute.'''

@measure_time
def my_function()-> None:
    time.sleep(2)

my_function()
