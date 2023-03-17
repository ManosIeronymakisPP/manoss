import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()
cursor.execute("""SELECT BillingCountry
                  FROM invoices
                  GROUP BY BillingCountry
                  ORDER BY SUM(Total) DESC
                  LIMIT 1
               """)
result = cursor.fetchall()
print("The country with the customers that spent the most is" ,result)
