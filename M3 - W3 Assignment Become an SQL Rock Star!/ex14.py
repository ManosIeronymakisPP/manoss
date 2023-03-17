import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  tracks.Name AS TrackName
FROM 
  tracks 
  JOIN invoice_items ON tracks.TrackId = invoice_items.TrackId 
  JOIN invoices ON invoice_items.InvoiceId = invoices.InvoiceId
WHERE 
  strftime('%Y', invoices.InvoiceDate) = '2010'
GROUP BY 
  TrackName
ORDER BY 
  sum(invoice_items.Quantity) DESC
LIMIT 1
               """)
result = cursor.fetchall()
print("The most purchased track of 2010 is" ,result)