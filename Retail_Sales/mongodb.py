# Step 2: Reading the data from MongoDB/NOSQL_DB

# Import the required libraries
import numpy as np
import pandas as pd
from pymongo import MongoClient                            # To connect with mongodb server
import time
from tkinter import *

# Creating a window using tkinter
window = Tk()
window.title('Data Pipeline - Retail dataset')  # Window Title

# Creating an label for Subheader
subheader1 = Label(window, text= '# Step 2: Data Reading from MongoDB #')
subheader1.grid(row=0, column=0, sticky=W)

# Creating an label for MongoDB Credentials
mongo_label = Label(window, text= 'Enter hostname and port number')
mongo_label.grid(row=1, column=0, sticky=W)

# Creating an Entry for the created label
mongo_host = StringVar()
mongo_entrybox = Entry(window, width = 30, textvariable = mongo_host)
mongo_entrybox.grid(row=1,column=1)

def server_conn():
    global mongo_hostname, client
    mongo_hostname = mongo_host.get()
    client = MongoClient(mongo_hostname)                     # mongodb://localhost:27017/
    print(client.list_database_names())                    
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=1, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a Button for created the label
mongo_button = Button(window, width = 15, text="hostname", command = server_conn)
mongo_button.grid(row=1,column=2)

# -------------------------------------------------------------------------------------
# Creating a label for selecting the database
dbselect_label = Label(window, text= 'Select the database')
dbselect_label.grid(row=2, column=0, sticky=W)

# Creating an entry for the created label
mongo_dbname = StringVar()
mongodb_entrybox = Entry(window, width = 30, textvariable = mongo_dbname)
mongodb_entrybox.grid(row=2,column=1)

def mongodb():
    global mongo_database, db
    mongo_database = str(mongo_dbname.get())
    db = client.mongo_database
    print(db.list_collection_names())
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=2, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a button for the created entry
dbname_button = Button(window, width = 15, text="database name", command = mongodb)
dbname_button.grid(row=2,column=2)

#-------------------------------------------------------------------------------------------

# Selecting the collection from the database

# Creating a label for selecting the collection
collection_label = Label(window, text= 'Select the collection')
collection_label.grid(row=3, column=0, sticky=W)

# Creating an entry for the created label
collection_name = StringVar()
collection_entrybox = Entry(window, width = 30, textvariable = collection_name)
collection_entrybox.grid(row=3,column=1)

def collection_detail():
    global data_collection
    db = client.faker_customer_data
    data_collection = db[collection_name.get()]
    #print(data_collection.find_one())
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=3, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a button for the created entry
collection_button = Button(window, width = 15, text="collection name", command = collection_detail)
collection_button.grid(row=3,column=2)

#-------------------------------------------------------------------------------------------

# Converting the data from collection to dataframe

# Creating a label for converting the collection data to dataframe
collection_label = Label(window, text= 'Display the size of the extracted dataset')
collection_label.grid(row=4, column=0, sticky=W)

# List to dataframe conversion
def mongo_size():
    global df1, size, cursor1, data_list
    # Creating a dataframe for the extracted records
    cursor1 = data_collection.find()
    data_list = list(cursor1)
    df1 = pd.DataFrame(data_list)
    size = df1.shape
    # Storing the dataframe locally
    df1.to_csv('DF2.csv', index=False)
    text1.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text1.insert(END, size) 

# Creating a text window for displaying the size of the dataset
text1 = Text(window, height=1.5, width=39)
text1.grid(row=4, column=1)

# Creating a button for the size
size_button = Button(window, width = 15, text="Dataset Size", command = mongo_size)
size_button.grid(row=4,column=2)

window.mainloop()
