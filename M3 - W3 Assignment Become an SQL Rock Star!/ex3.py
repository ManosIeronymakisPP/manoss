import sqlite3
import os



os.chdir("C:\\Users\\User\\Desktop\\SQL module")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\SQL module\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT customers.FirstName || ' ' || customers.LastName AS CustomerName,
       invoices.InvoiceId,
       invoices.InvoiceDate,
       invoices.BillingCountry
                 FROM invoices
                 JOIN customers
                 ON invoices.CustomerId = customers.CustomerId
                 WHERE invoices.BillingCountry = 'USA'
               """)
result = cursor.fetchall()

print(result)
