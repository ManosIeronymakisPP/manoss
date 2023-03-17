import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  employees.FirstName || ' ' || employees.LastName AS SalesAgentName, 
  sum(invoice_items.UnitPrice * invoice_items.Quantity) AS TotalSales
FROM 
  employees 
  JOIN customers ON employees.EmployeeId = customers.SupportRepId 
  JOIN invoices ON customers.CustomerId = invoices.CustomerId 
  JOIN invoice_items ON invoices.InvoiceId = invoice_items.InvoiceId 
GROUP BY 
  SalesAgentName
               """)          
result = cursor.fetchall()      
print(result)
