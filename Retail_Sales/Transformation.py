# Data Pre-processing / Data Transformation

# Import the required libraries
import numpy as np
import pandas as pd
import time
from tkinter import *

# Creating a window using tkinter
window = Tk()
window.title('Data Pipeline - Retail dataset')  # Window Title

#----------------------------------------------------------------------------------------------------------------------------

# Creating an label for Subheader
subheader1 = Label(window, text= '---# Step 3: Transform / Preprocessing #---')
subheader1.grid(row=0, column=0, sticky=W)

# ---------------------------------------------------------------------------------------------------------------------------

# Creating an label for importing first dataset fetched from MYSQL_DB
data1_label = Label(window, text= 'Import the data fetched from MYSQL_DB:')
data1_label.grid(row=1, column=0, sticky=W)

# Creating an Entry for the created label
data1 = StringVar()
data1_entrybox = Entry(window, textvariable = data1)
data1_entrybox.grid(row=1,column=1)

def import_data1():
    global DF1
    # Read the data as dataframe
    DF1 = pd.read_csv(data1.get())
    print(DF1.head())
    print(DF1.shape)                   
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=1, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a Button for created the label
data1_button = Button(window, width = 15, text="DataFrame-1", command = import_data1)
data1_button.grid(row=1,column=2)

# -------------------------------------------------------------------------------------

# Creating an label for second dataset fetched from Mongo_DB
data2_label = Label(window, text= 'Import the data fetched from Mongo_DB:')
data2_label.grid(row=2, column=0, sticky=W)

# Creating an Entry for the created label
data2 = StringVar()
data2_entrybox = Entry(window, textvariable = data2)
data2_entrybox.grid(row=2,column=1)

def import_data2():
    global DF2
    # Read the data as dataframe
    DF2 = pd.read_csv(data2.get())
    print(DF2.head())
    print(DF2.shape)                   
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=2, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a Button for created the label
data2_button = Button(window, width = 15, text="DataFrame-2", command = import_data2)
data2_button.grid(row=2,column=2)

# -------------------------------------------------------------------------------------

# Creating an label for merging the dataset
merge_label = Label(window, text= 'Merge Datframe-1 and Dataframe-2')
merge_label.grid(row=3, column=0, sticky=W)

def merge_data():
    global DF, shape
    # Merge the dataframe using pandas library
    DF = pd.merge(DF1, DF2, on = 'id')
    print(DF.head())
    print(DF.shape)
    shape = (DF1.shape, DF2.shape, DF.shape)
    text.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text.insert(END, shape)                    
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=3, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a text window for displaying the shape of the merged dataframe
text = Text(window, height=1.5, width=20)
text.grid(row=3, column=2)

# Creating a Button for created the label
merge_button = Button(window, text="Merge DataFrames", command = merge_data)
merge_button.grid(row=3,column=1)

# -------------------------------------------------------------------------------------
# Preprocessing - 1

# Creating an label for cheking null values
preprocess1_label = Label(window, text= 'Drop unwnated columns and Check for null values, duplicates')
preprocess1_label.grid(row=4, column=0, sticky=W)

def preprocess1_data():
    global null, null_value, duplicates, output, shape1
    # Merge the dataframe using pandas library
    DF.drop(columns = ['Item_Identifier_y', '_id'], axis = 1, inplace = True)
    shape1 = DF.shape
    null = DF.isnull().sum()
    j = []
    for i in null:
        if i > 0:
            j.append(i)
    null_value = len(j)
    duplicates = len(DF[DF.duplicated()])
    print(duplicates)
    output = (DF.shape, null_value, duplicates)
    text1.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text1.insert(END, output)                    
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=4, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a text window for displaying the shape of the merged dataframe
text1 = Text(window, height=1.5, width=20)
text1.grid(row=4, column=2)

# Creating a Button for created the label
preprocess1_button = Button(window, text="Preprocess-1", command = preprocess1_data)
preprocess1_button.grid(row=4,column=1)

# -------------------------------------------------------------------------------------
# Preprocessing - 2

# Creating a label for selecting the method of replacement for null values
null_label = Label(window, text= 'Method to replace null values - mean, median, mode, drop')
null_label.grid(row=5, column=0, sticky=W)

# Creating an Entry for the created label
null_replacement = StringVar()
null_entrybox = Entry(window, textvariable = null_replacement)
null_entrybox.grid(row=5,column=1)

def null_method():
    global method
    # Read the data as dataframe
    method = null_replacement.get()                  
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=5, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a Button for created the label
null_button = Button(window, width = 15, text="Replacing Method", command = null_method)
null_button.grid(row=5,column=2)

#-----------------------

# Creating a label for removing null and duplicate values
preprocess2_label = Label(window, text= 'Null Values & Duplicates Treatment')
preprocess2_label.grid(row=6, column=0, sticky=W)

def preprocess2_data():
    global result, shape2, null, null_value, duplicates
    # Merge the dataframe using pandas library
    DF.drop_duplicates(keep="first", inplace=True)
    shape2 = DF.shape

    if method == 'median':
        DF.fillna(DF.median(), inplace = True)
        result = 'Replaced'
    elif method == 'mean':
        DF.fillna(DF.mean(), inplace = True)
        result = 'Replaced'
    elif method == 'mode':
        DF.fillna(DF.mode(), inplace = True)
        result = 'Replaced'
    elif method == 'drop':
        DF.dropna(inplace = True)
        result = 'Dropped'
    else:
        result = 'Choose only from the given methods'
    print(DF.isnull().sum())
    null = DF.isnull().sum()
    j = []
    for i in null:
        if i > 0:
            j.append(i)
    null_value = len(j)
    duplicates = len(DF[DF.duplicated()])
    DF.to_csv('DF.csv', index=False)
    output = (DF.shape, null_value, duplicates, result)
    text2.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text2.insert(END, output)                    
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=6, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a text window for displaying the shape of the merged dataframe
text2 = Text(window, height=1.5, width=20)
text2.grid(row=6, column=2)

# Creating a Button for created the label
preprocess2_button = Button(window, text="Preprocess-2", command = preprocess2_data)
preprocess2_button.grid(row=6,column=1)

# -------------------------------------------------------------------------------------
# Preprocess-3 Visualization

# import necessary libraries
import re          
import pandas_profiling
from autoviz.AutoViz_Class import AutoViz_Class
import sweetviz as sv

# Creating a label for getting the reports in the form of visualization
preprocess3_label = Label(window, text= 'Generate report for cleaned dataset using visualization library')
preprocess3_label.grid(row=7, column=0, sticky=W)

def preprocess3_data():
    global profile_pandas, profile_sweetviz, profile_autoviz, report

    # Pandas profiling
    profile_pandas = pandas_profiling.ProfileReport(DF)
    profile_pandas.to_file('pandas_profile.html')

    # Sweetviz Implementation
    profile_sweetviz = sv.analyze(DF)
    profile_sweetviz.show_html('sweetviz.html')

    # Autoviz Implementation
    AV = AutoViz_Class()
    profile_autoviz = AV.AutoViz('DF.csv', sep=',', depVar='', dfte=None, header=0, verbose=1, lowess=False,
               chart_format='html',max_rows_analyzed=150000,max_cols_analyzed=30, save_plot_dir=None)

    report = 'Report generated'
    text3.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text3.insert(END, report)                    
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=7, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a text window for displaying the shape of the merged dataframe
text3 = Text(window, height=1.5, width=20)
text3.grid(row=7, column=2)

# Creating a Button for created the label
preprocess3_button = Button(window, text="Preprocess-3", command = preprocess3_data)
preprocess3_button.grid(row=7,column=1)

# -------------------------------------------------------------------------------------

# Creating an label for Subheader
subheader2 = Label(window, text= '---# Step 4: Data Versioning #---')
subheader2.grid(row=8, column=0, sticky=W)

# -------------------------------------------------------------------------------------
# Data Version

# import necessary libraries
from datetime import date, time, datetime

# Creating a label for getting the reports in the form of visualization
dataver_label = Label(window, text= 'Enter the path where to store the final cleansed dataset')
dataver_label.grid(row=9, column=0, sticky=W)

# Creating an Entry for the created label
dataversion = StringVar()
dataver_entrybox = Entry(window, textvariable = dataversion)
dataver_entrybox.grid(row=9,column=1)


def data_version():
    global file_path, curr_datetime, path
    # This will get the path for the dataset where to store it everytime
    file_path = dataversion.get()
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')              # Setting the date and time format
    path = str(file_path) + curr_datetime + 'Final_Data.csv'                       # Merging the file_path with current date and time
    DF.to_csv(path, index=False)                                              # Saving the dataframe with the added datetime extension
               
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=9, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a Button for created the label
dataver_button = Button(window, width = 15, text="Data-versioning", command = data_version)
dataver_button.grid(row=9,column=2)

# -------------------------------------------------------------------------------------
# Pushing the data to MongoDB

# Getting the name of the database from the user
# Import the necessary modules
from pymongo import MongoClient
import json

# Creating a label for storing the cleaned dataset in NOSQL database
mongdb_label = Label(window, text= 'Push the final data to a database')
mongdb_label.grid(row=10, column=0, sticky=W)

def mongdb():
    global mong_database, db1, client, retail_rec, result
    # Connect to MongoDB and icecream_data database
    try:
        client = MongoClient('localhost',27017)               # Make sure that your mongodb server is running
        db1 = client.retail_sales
        print("Connected successfully!")
    except:  
        print("Could not connect to MongoDB")

    # Collection_name
    retail_rec = DF.to_dict(orient='records')                       # Converting the dataframe tpo dictionary
    try:
        rec_id = db1.retail_info.insert_many(retail_rec)
        print("Data inserted with record ids", rec_id)
    except:
        print("Could not insert into MongoDB")
    result = 'Data is inserted'

    text4.delete('1.0', END)                                 # Deleting the existing entry in the text window
    text4.insert(END, result)                    
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=10, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a text window for displaying the shape of the merged dataframe
text4 = Text(window, height=1.5, width=20)
text4.grid(row=10, column=2)
    
# Creating a button for the created entry
mongdb_button = Button(window, text="Data_Storage", command = mongdb)
mongdb_button.grid(row=10,column=1)


window.mainloop()