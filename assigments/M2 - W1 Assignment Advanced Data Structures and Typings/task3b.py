# typing is a standard library
from typing import List, Dict


def GetData(name: str, id: int, degree: float, startyear: int, endyear: str) -> List[Dict[str, int]]: 
	return [{"name": name, "id": id, "degree":degree, "startyear":startyear, "endyear":endyear}]

print(GetData("manos", 100,9,2014,"2019")) #data1
print(GetData("chrisa", 101,9,2015,"2020")) #data2
print(GetData("mike", 102,9,2019,"2022"))  #data3
print(GetData("stavros", 103,9,2021,"not yet")) #data4


"we created an example with 4 data.Of course the user can put as many data she/he wants"
"our typing structure inside the function Getdata has str for the name and endyear, int for the id, startyear and float for degree. Our function is returning a dictionary inside a list."

#############################################EXTRA#################################################################
#THIS IS ANOTHER WAY TO VISUALISE A BIT NICER OUR DATA IN OUR EXAMPLE WITHOUT TO COMPLEX CODING


"lets see our stucture with an example of 4 people"
"every time the user should give a dictionary with our data manually and every time we add a loop to read those data"



dict1={"name:": "manos","id:": "100","degree:": "9","startyear:": "2014","endyear:" : "2019"}
dict2={"name:": "chrisa","id:": "101","degree:": "8","startyear:": "2015","endyear:" : "2020"}
dict3={"name:": "mike","id:": "102","degree:": "7","startyear:": "2021","endyear:" : "2022"}
dict4={"name:": "stavros","id:": "103","degree:": "10","startyear:": "2019","endyear:" : "not yet"}


for x, y in dict1.items():
  print(x,y)
print()
for x, y in dict2.items():
  print(x,y)
print()
for x, y in dict3.items():
  print(x,y)
print()
for x, y in dict4.items():
  print(x,y)
