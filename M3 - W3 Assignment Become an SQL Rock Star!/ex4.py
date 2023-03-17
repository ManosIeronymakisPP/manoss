import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT COUNT(*)
               FROM employees
               WHERE Title = 'Sales Support Agent';
               """)
result = cursor.fetchall()
print("The employees that working as Sales Agents are", result)
