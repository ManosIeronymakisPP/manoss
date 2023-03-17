import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  employees.FirstName || ' ' || employees.LastName AS SalesAgentName
FROM 
  employees 
  JOIN customers ON employees.EmployeeId = customers.SupportRepId 
  JOIN invoices ON customers.CustomerId = invoices.CustomerId 
  JOIN invoice_items ON invoices.InvoiceId = invoice_items.InvoiceId 
WHERE 
  strftime('%Y', invoices.InvoiceDate) = '2012'
GROUP BY 
  SalesAgentName
ORDER BY 
  sum(invoice_items.UnitPrice * invoice_items.Quantity) DESC
LIMIT 1
               """)
result = cursor.fetchall()
print("The sales agent that made the most sales in 2012 is" ,result)
