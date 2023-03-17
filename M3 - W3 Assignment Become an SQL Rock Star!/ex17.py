import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  media_types.Name AS MediaType
FROM 
  media_types 
  JOIN tracks ON media_types.MediaTypeId = tracks.MediaTypeId 
  JOIN invoice_items ON tracks.TrackId = invoice_items.TrackId 
GROUP BY 
  MediaType
ORDER BY 
  sum(invoice_items.Quantity) DESC
LIMIT 1
               """)
result = cursor.fetchall()
print("The most purchased Media Type is" ,result)