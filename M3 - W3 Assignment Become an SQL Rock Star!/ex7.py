import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT BillingCountry, COUNT(InvoiceId) AS Invoices
                  FROM invoices
                  GROUP BY BillingCountry
               """)
result = cursor.fetchall()
print("The invoices that created for each country are", result)