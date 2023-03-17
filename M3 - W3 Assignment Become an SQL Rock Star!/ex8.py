import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT PlaylistId, COUNT(TrackId) AS Tracks
                  FROM playlist_track
                  GROUP BY PlaylistId
               """)
result = cursor.fetchall()
print("The tracks that are in each playlist are", result)