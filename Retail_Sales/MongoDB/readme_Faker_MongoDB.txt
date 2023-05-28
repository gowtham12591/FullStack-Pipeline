### Pushing the data from Faker to Mongo DB using Apache Kafka  - (Windows Machine)

Step 1: Installing Faker

    - To create the data we are using faker library here
    - Install faker using '!pip install faker' if not done

Step 2: Initializing Apache-Kafka & Creating a topic      

    - To stream the data kafka is utilized, Install kafka using 
        pip install kafka-python
    - Apart from this Apache-Kafka has to be installed seperately in the system. Please refer to the below link for easy installation,
        - https://kafka.apache.org/downloads
        - https://medium.com/@sangeethaprabhagaran/creating-a-kafka-topic-in-windows-e51b15e5ccd4 
    - Once kafka is installed, zookeeper and kafka-server has to be started
    - Navigate to the place where kafka folder is saved in command prompt
	  - cd C:\kafka\kafka_2.12-3.4.0
    - Start zookeeper and kafka-server using
        - .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
        - .\bin\windows\kafka-server-start.bat .\config\server.properties
    - Create a topic using,
        - bin\windows\kafka-topics.bat --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic customer-details-faker
        - Here the topic name chosen is 'customer-details-faker' with 1 partition and 1 replication factor**, replication factor and partition can be any value based on the need.

Step 3: Stream the data from Jupyter notebook to Kafka topic
    - Refer the code in faker_customer_producer.py
    - Open command prompt and navigate to the folder where faker_producer.py is saved and run the below command,
        python faker_customer_producer.py

Step 4: MongoDB installation
    - Install pymongo using '!pip install pymongo'
    - Apart from this MongoDB has to be installed seperately, please refer to the below official page for downloading the software
    - https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows-unattended/
    - If you are finding it difficult please refer to some youtube videos for installation and how to turn on the mongodb server
    - Open MongoDB compass and connect it to the server 

Step 5: Push the data from Kafka topic to MongoDB
    - Refer the code in faker_customer_consumer.py
    - Open command prompt and navigate to the folder where faker_consumer.py is saved and run the below command,
        python faker_customer_consumer.py

Step 6: Check for database in MongoDB
    - Open MongoDB compass and check for the database named faker_bikeprice_data and check for the collection faker_bikeprice_info
    - Open the collection in a seperate tab and verify the data with the MYSQL database
