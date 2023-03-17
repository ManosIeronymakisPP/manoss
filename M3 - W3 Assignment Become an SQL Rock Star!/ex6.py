import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  strftime('%Y', InvoiceDate) AS Year, 
  COUNT(*) AS Invoices
FROM invoices 
WHERE InvoiceDate BETWEEN '2010-01-01' AND '2011-12-31' 
GROUP BY Year;
               """)
result = cursor.fetchall()
print("The invoices in 2010 and 2011 are", result)

cursor.execute("""SELECT 
  strftime('%Y', InvoiceDate) AS Year, 
  SUM(Total) AS TotalSales 
FROM invoices 
WHERE InvoiceDate BETWEEN '2010-01-01' AND '2011-12-31' 
GROUP BY Year;
               """)             
result = cursor.fetchall()
print("The respective total sales for each of those years are", result)