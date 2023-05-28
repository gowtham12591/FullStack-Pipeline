

### Step 1: Data Reading from MYSQL_DB / RDBMS

# Import the required libraries
import numpy as np
import pandas as pd
import mysql.connector                              # To connect with mysql server
import time
from tkinter import *

# Creating a window using tkinter
window = Tk()
window.title('Full Stack Case Pipeline - Retail dataset')  # Window Title

# Creating an label for MYSQL Database Credentials
subheader1 = Label(window, width = 35, text= '# Step 1: Data Reading from MYSQL DB #')
subheader1.grid(row=0, column=0, sticky=W)

# Creating an label for MYSQL Database Credentials
host_label = Label(window, width = 30, text= 'Enter hostname')
host_label.grid(row=1, column=0, sticky=W)

db_label = Label(window, width = 30, text= 'Enter Database Name')
db_label.grid(row=2, column=0, sticky=W)

user_label = Label(window, width = 30, text= 'Enter Username')
user_label.grid(row=3, column=0, sticky=W)

pass_label = Label(window, width = 30, text= 'Enter Password')
pass_label.grid(row=4, column=0, sticky=W)

# Creating an Entry for all the created label
hostname = StringVar()
host_entrybox = Entry(window, width = 20, textvariable = hostname)
host_entrybox.grid(row=1,column=1)

dbname = StringVar()
db_entrybox = Entry(window, width = 20, textvariable = dbname)
db_entrybox.grid(row=2,column=1)

username = StringVar()
user_entrybox = Entry(window, width = 20, textvariable = username)
user_entrybox.grid(row=3,column=1)

password = StringVar()
pass_entrybox = Entry(window, width = 20, textvariable = password)
pass_entrybox.grid(row=4,column=1)

def host_name():
    global host
    # Getting the user credentials for MYSQL Database
    host = hostname.get()                           #'localhost'
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=1, column=3)
    Confirm_entrybox.insert(1, str(confirm))                        

def db():
    global database
    database = dbname.get()
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=2, column=3)
    Confirm_entrybox.insert(1, str(confirm))

def user_name():
    global user
    user = username.get()
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=3, column=3)
    Confirm_entrybox.insert(1, str(confirm))

def passw():
    global p_word
    p_word = password.get()
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=4, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a Button for all the labels
host_button = Button(window, width = 15, text="hostname", command = host_name)
host_button.grid(row=1,column=2)

db_button = Button(window, width = 15, text="database name", command = db)
db_button.grid(row=2,column=2)

user_button = Button(window, width = 15, text="user name", command = user_name)
user_button.grid(row=3,column=2)

pass_button = Button(window, width = 15, text="password", command = passw)
pass_button.grid(row=4,column=2)


# Creating a label for fetching all the records
rec_label = Label(window, width = 30, text= 'Select the data from the table')
rec_label.grid(row=5, column=0, sticky=W)

# Creating an entry for the created label records
fetching_rec = StringVar()
rec_entrybox = Entry(window, width = 20, textvariable = fetching_rec)
rec_entrybox.grid(row=5,column=1)                                    

def fetch_records():
    global cursor, record, connection

    # Mysql database connection with jupyter notebook
    connection = mysql.connector.connect(
                                    host = host, 
                                    database = database,
                                    user = user,
                                    password = p_word)    
    # Set the cursor to fetch all the records from the bike_price table
    cursor = connection.cursor()
    cursor.execute(fetching_rec.get())                                   # 'select * from  retail_sales'
    record = cursor.fetchall()
    cursor.close()
    connection.close()

    # Confrim entry for the 'select statement'
    confirm = 'Done'
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=5, column=3)
    Confirm_entrybox.insert(1, str(confirm))


# Creating a button for the records fetched
rec_button = Button(window, width = 15, text="Fetching records", command = fetch_records)
rec_button.grid(row=5,column=2)

# Creating an label for extracted data
dataset_size = Label(window, width = 30, text= 'Display the size of the extracted dataset')
dataset_size.grid(row=6, column=0, sticky=W)

# Dataset size
def size():
    global df, size
    # Creating a dataframe for the extracted records
    df = pd.DataFrame(record, columns = ['id', 'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
                                     'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
                                     'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales'])
    size = df.shape
    text1.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text1.insert(END, size) 

# Creating a text window for displaying the size of the dataset
text1 = Text(window, height=1, width=15)
text1.grid(row=6, column=1)

# Creating a button for the size
size_button = Button(window, width = 15, text="Dataset Size", command = size)
size_button.grid(row=6,column=2)

window.mainloop()


