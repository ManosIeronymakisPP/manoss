import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT DISTINCT BillingCountry
                  FROM invoices
               """)              
result = cursor.fetchall()      
print("The list of all countries that the invoices send is",result)