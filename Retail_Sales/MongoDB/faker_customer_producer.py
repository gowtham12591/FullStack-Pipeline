# Customer Dataset Creation

# Please read the readme file for initial set-up

# Import the below libraries
import random                               # For random genration of data
from faker.providers import BaseProvider    # Helps to provide fake data 
from faker import Faker
import json
from kafka import KafkaProducer
import time
import random
import argparse
import numpy as np
import pandas as pd

#Note: Fake data is useful for someone who is learning to process data and pipelining data from one system to another.

df = pd.read_csv('Retail_Sales_Data.csv')                       # Reading the Retails_sales_data
id = df.Item_Identifier.unique()                                # Getting the unique ids alone to create fake customer details using this id as reference

# - Adding Customer_details data with a method:
#   * item_id for selecting the particular id of the item corresponding to the customer detail generated.

# ----------------------------------------------------------------------------------------------------------------------------------------
# The class that is used here is the source data which will be randomly selected when called

class Customer_details(BaseProvider):                           # Defining a base class
# Some top brands of bike
    def item_id(self):
        item_ids = id                                           # id is the 'Item_Identifier' which is used for creating fake data of customers
        return item_ids[random.randint(0, len(item_ids)-1)]     # Returning a random id from the list
    
# ---------------------------------------------------------------------------------------------------------------------------------------
# Creating a Faker instance and seeding to have the same results every time we execute the script
fake = Faker()
Faker.seed(42)

# Adding the newly created Customer_details to the Faker instance
fake.add_provider(Customer_details)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Creating function to generate the Customer_details

def produce_Customer_details(ordercount = 1):
    # message composition
    message = {
        'id': ordercount,                                   # Auto increment by i=i+1
        'name': fake.unique.name(),                         # Fake name from the faker library
        'phoneNumber': fake.unique.phone_number(),          # Fake number form the faker library
        'address': fake.address(),                          # Fake address form the faker library
        'Item_Identifier': fake.item_id()                   # Mapping item_id to the item_identifier             
    }
    key = {'brand': fake.item_id()}
    return message, key                                     # Returning as key value pair - better for storing in NoSQL Database

# --------------------------------------------------------------------------------------------------------------------------------------
# Pusing message from Jupyter to Kafka-topic

# produce_msgs function starts producing messages with Faker
def produce_msgs(hostname='localhost',
                 port='9092',
                 topic_name='customer-details-faker',       # Name of the topic created in kafka
                 nr_messages=13990,                         # Number of messages to produce (0 represents unlimited)
                 max_waiting_time_in_sec=0.1):
    
    # Function for Kafka Producer with certain settings related to the Kafka's Server
    producer = KafkaProducer(
                bootstrap_servers=hostname+":"+port,
                value_serializer=lambda v: json.dumps(v).encode('ascii'),               
                key_serializer=lambda v: json.dumps(v).encode('ascii')
                )
    
    # When the number of messages are 0 or less then it is defined as infinite
    if nr_messages <= 0:
        nr_messages = float('inf')
    
    i = 0                                                   # Setting the initial number of orders to be zero
    while i < nr_messages:
        message, key = produce_Customer_details(i)          # Getting the key and message from the function which is used for generating the bike order

        print("Sending: {}".format(message))
        # sending the message to Kafka
        producer.send(topic_name,
                      key=key,
                      value=message)
        
        # Sleeping time / Waiting time
        sleep_time = random.randint(0, max_waiting_time_in_sec * 10)/10
        print("Sleeping for..."+str(sleep_time)+'s')
        time.sleep(sleep_time)

        # Force flushing of all messages
        if (i % 100) == 0:
            producer.flush()
        i = i + 1
    producer.flush()

# calling the main produce_msgs function: parameters are:
#   * nr_messages: number of messages to produce
#   * max_waiting_time_in_sec: maximum waiting time in sec between messages

produce_msgs()

#----------------------------------------------------------------------------------------------------------------------------------------
