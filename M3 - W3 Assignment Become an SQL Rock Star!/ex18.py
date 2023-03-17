import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  invoices.InvoiceId, 
  sum(invoice_items.Quantity) AS TotalTracksPurchased
FROM 
  invoices
  JOIN invoice_items ON invoices.InvoiceId = invoice_items.InvoiceId
  JOIN tracks ON invoice_items.TrackId = tracks.TrackId
GROUP BY 
  invoices.InvoiceId
HAVING 
  count(distinct tracks.GenreId) > 1
              """)
result = cursor.fetchall()
print("The number of tracks purchased in all invoices that contain more than one genre is (invoice id, number of tracks in that invoice):" ,result)