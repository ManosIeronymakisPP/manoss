import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT customers.FirstName || ' ' || customers.LastName AS CustomerName
               FROM customers
               WHERE country = 'Brazil'
               """)
result = cursor.fetchall()
print("The customers from the Brazil are", result)
cursor.execute("""SELECT COUNT(*)
               FROM customers
               WHERE country = 'Brazil'
               """)
result = cursor.fetchall()
print("The Brazilian customers are", result)
