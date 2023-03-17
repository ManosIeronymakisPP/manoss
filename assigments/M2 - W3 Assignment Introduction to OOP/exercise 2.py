class Person:
    def __init__(self, name:str, number_of_legs= 2 ,is_alive= True):
        self.name = name
        self.number_of_legs = number_of_legs
        self.is_alive = is_alive


class Student(Person):
    def __init__(self, name:str, grade:str, number:str)->str:
        Person.__init__(self,name, number_of_legs= 2 ,is_alive= True)

class Lecture(Person):
    def __init__(self, name:str, Numberln_Step2:str, Numberln_step3:str)->str:
        Person.__init__(self,name, number_of_legs= 2 ,is_alive= True)
