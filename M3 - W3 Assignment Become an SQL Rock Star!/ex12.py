import sqlite3
import os
os.chdir("C:\\Users\\User\\Desktop")
filepath =  "chinook.db"
conn = sqlite3.connect("C:\\Users\\User\\Desktop\\chinook.db")
cursor = conn.cursor()

cursor.execute("""SELECT 
  employees.FirstName || ' ' || employees.LastName AS SalesAgentName, 
  count(*) AS NumberOfCustomers
FROM 
  employees 
  JOIN customers ON employees.EmployeeId = customers.SupportRepId 
GROUP BY 
  SalesAgentName
ORDER BY 
  NumberOfCustomers DESC

               """)
result = cursor.fetchall()
print("The number of customers assigned to each sales agent is" ,result)