import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  artists.Name AS ArtistName
FROM 
  artists 
  JOIN albums ON artists.ArtistId = albums.ArtistId 
  JOIN tracks ON albums.AlbumId = tracks.AlbumId 
  JOIN invoice_items ON tracks.TrackId = invoice_items.TrackId 
GROUP BY 
  ArtistName
ORDER BY 
  sum(invoice_items.Quantity) DESC
LIMIT 3
               """)
result = cursor.fetchall()
print("The top 3 best selling artists are" ,result)
