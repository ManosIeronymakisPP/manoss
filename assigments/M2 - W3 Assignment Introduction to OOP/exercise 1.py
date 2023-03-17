class Book:
    #constructor
    def __init__(self:str, title:str, first:str, last:str, price:str)->str:
        self.title = title
        self.first = first
        self.last = last
        self.price = price
        
    #view method
    def view(self:str)->str:
        return "The title of the book is " + self.title + "."+"The authors full name is "+ self.first +" "+ self.last + " and it costs "+ self.price +"$."

book1 = Book("'The design of everyday things'","Don","Norman", "19")

print(book1.view())
