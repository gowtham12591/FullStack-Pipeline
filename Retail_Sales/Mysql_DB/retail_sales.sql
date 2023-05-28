# Create a database on Retail Sales and use it going forward
drop database if exists retail_sales_db;
create database retail_sales_db;
use retail_sales_db;

-- ---------------------------------------------------------------------------------------------
####### Table: Bike Price #######
create table retail_sales(
	id int primary key auto_increment,
    Item_Identifier varchar(50),
    Item_Weight float DEFAULT NULL,
    Item_Fat_Content varchar(50) DEFAULT NULL,
    Item_Visibility float DEFAULT NULL,
    Item_Type varchar(50),
    Item_MRP float DEFAULT NULL,
    Outlet_Identifier varchar(50),
    Outlet_Establishment_Year int DEFAULT NULL,
    Outlet_Size varchar(50),
    Outlet_Location_Type varchar(50),
    Outlet_Type varchar(50),
    Item_Outlet_Sales float DEFAULT NULL
    );

# Show the table details    
desc retail_sales;

# Check the details of the table
select * from retail_sales;

# Checking whether all the data has been imported
select count(id) from retail_sales; 