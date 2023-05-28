# Data Consumed from kafka-topic to MongoDB

# Import the necessary modules
from kafka import KafkaConsumer
from pymongo import MongoClient
import json

# Connect to MongoDB and create faker_customer_data database
try:
   client = MongoClient('localhost',27017)                      # Make sure that your mongodb server is running
   db = client.faker_customer_data
   print("Connected successfully!")
except:  
   print("Could not connect to MongoDB")
    
# connect kafka consumer to desired kafka topic	
consumer = KafkaConsumer('customer-details-faker',bootstrap_servers=['localhost:9092'])

# ----------------------------------------------------------------------------------------------------------------------------------------

# Parse received data from Kafka
for msg in consumer:
   #print('message:', msg)
   record = json.loads(msg.value)
   id = record['id']
   name = record['name']
   phoneNumber = record['phoneNumber']
   address = record['address']
   Item_Identifier = record['Item_Identifier']
    
    # Create dictionary and ingest data into MongoDB
   try:
      customer_rec = {'id': id, 'name':name, 'phoneNumber':phoneNumber, 'address':address, 'Item_Identifier': Item_Identifier}
      print('customer_rec:', customer_rec)
      rec_id = db.faker_customer_info.insert_one(customer_rec)
      print("Data inserted with record ids", rec_id)
   except:
      print("Could not insert into MongoDB")