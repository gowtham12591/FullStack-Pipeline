

#----------------------------------------------------------------------------------------------------------------------------------------
# AutoML - Creating a block for Automatic Model Training and Testing
# Library

from tkinter import *
import pandas as pd
import re

# App window
window = Tk()
window.title(' Regression GUI - Auto ML Prediction ')  # Window Title

#----------------------------------------------------------------------------------------------------------------------------------------
# Creating an label for Subheader
subheader1 = Label(window, text= '---# Step 4: Auto ML #---')
subheader1.grid(row=0, column=0, sticky=W)

#----------------------------------------------------------------------------------------------------------------------------------------
# Import the dataset name along with its location to import 

Name=Label(window, text = "Enter file Name along with its path ")
Name.grid(row=1,column=0,sticky=W)

Name_var= StringVar()
Name_entrybox= Entry(window, width = 30, textvariable=Name_var)
Name_entrybox.grid(row=1, column=1)

def Import_Data():
    global DF
    DF_Name=Name_var.get()
    DS_extension=re.findall("\..*", DF_Name) 
    if DS_extension==['.xlsx']:
        DF = pd.read_excel(DF_Name)
    elif DS_extension==['.csv']:
        DF = pd.read_csv(DF_Name)
    DF.head()
    DF.drop(['name', 'phoneNumber', 'address'], axis=1, inplace=True)
    # Blank empty window to print confirmation
    confirm="Done"
    Confirm_entrybox = Entry(window,width=16)
    Confirm_entrybox.grid(row=1, column=3)
    Confirm_entrybox.insert(1, str(confirm))   

Import_Data_Button = Button(window, width = 15, text="Import Data", command=Import_Data)
Import_Data_Button.grid(row=1,column=2)

#-------------------------------------------------------------------------------------------------------------------------------
# Get the Target variable name and split the dataset to train and test

Target = Label(window, text="Target Column")
Target.grid(row=2,column=0,sticky=W)

Target_var = StringVar()
Target_entrybox = Entry(window, width=30, textvariable=Target_var)
Target_entrybox.grid(row=2, column=1)

def Target_Data():
    global DF, X, y, Target_Name, X_train, X_test, y_train, y_test
    Target_Name = Target_var.get()
    # Get the column names from the dataframe
    Column_name = DF.columns
    print(Column_name)
    found=0
    # Loop condition to check the whether target name is there in the dataset or not and then splitting the dataset
    for i in range(len(Column_name)):
        #print(i)
        if Column_name[i] == Target_Name:
            confirm="Found"
            y = DF[Target_Name]                 # Target variable
            X = DF.drop([Target_Name], axis=1)    # Predictor variable
            print(X.columns)
            #X = X.drop(['name'], axis = 1)
            # Train and text split
            from sklearn.model_selection import train_test_split
            X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True) # Train-val splitting
            X_test, X_test1, y_test, y_test1 = train_test_split(x_val, y_val, test_size=0.10, random_state=42, shuffle=True)
            X_test1.to_csv('test_data.csv')

            break

        else:
            confirm="Not Found"
    
    Confirm_entrybox = Entry(window, width=10)
    Confirm_entrybox.grid(row=2, column=3)
    Confirm_entrybox.insert(1, str(confirm))

# Creating a button for the created label
Target_Button = Button(window, width = 15, text="Target Column", command=Target_Data)
Target_Button.grid(row=2, column=2)

# -------------------------------------------------------------------------------------
# Encoding the non-numerical features

# import necessary libraries
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() 

# Creating a label for converting the non-numerical features to numerival features
preprocess4_label = Label(window, text= 'Encode Non-Numerical Features')
preprocess4_label.grid(row=3, column=0, sticky=W)

def Encode():
    global encode
    # Encode both train and test data seperately
    for i in DF.columns:
        if DF[i].dtypes == 'object':
            X_train[i] = label_encoder.fit_transform(X_train[i])
            X_test[i] = label_encoder.transform(X_test[i]) 
    print(X_train.isnull().sum())
    X_test.to_csv('test_data.csv', index=False)
             
    # Blank empty window to print confirmation
    confirm = "Done"
    Confirm_entrybox = Entry(window, width=16)
    Confirm_entrybox.grid(row=3, column=2)
    Confirm_entrybox.insert(1, str(confirm))


# Creating a Button for created the label
preprocess4_button = Button(window, width = 15, text="Encode", command = Encode)
preprocess4_button.grid(row=3,column=1)

#-------------------------------------------------------------------------------------------------------------------------------
# Model Training

# Creating an label for Subheader
Modelling = Label(window, text= " ---# Step 5: Training Supervised learning Models #---")
Modelling.grid(row=4, column=0, sticky = W)

#------------------------------------------------------------------------------

# 1. Linear regression -------------------------------------------------------
# Import required libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

LR = Label(window, text="Linear Regression")
LR.grid(row=5, column=0, sticky=E)


def LR():
    global LR_model, y_pred, LR_rmse_Train, LR_rmse_Test, LR_Tr, LR_Te
    
    # Logistic Regression
    LR_model = LinearRegression()
    LR_model.fit(X_train, y_train)
    y_pred = LR_model.predict(X_train)
    LR_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    LR_r2score_Train = round(r2_score(y_train, y_pred), 2)
    LR_Tr, LR_Tr_r2 = str(LR_rmse_Train), str(LR_r2score_Train)

    y_pred = LR_model.predict(X_test)
    LR_rmse_Test = round(mean_squared_error(y_test,y_pred), 2)
    LR_r2score_Test = round(r2_score(y_test, y_pred), 2)
    LR_Te, LR_Te_r2 = str(LR_rmse_Test) , str(LR_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=5, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + LR_Tr + ', ' + LR_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=5, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + LR_Te + ', ' + LR_Te_r2))

LR_Button = Button(window, text="RUN", command=LR)
LR_Button.grid(row=5, column=1)

#--------------------------------------------------------------------------------------------------------------------------------
# 2. SVM
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

SVM = Label(window, text="Support Vector Regressor")
SVM.grid(row=6, column=0, sticky=E)

SVM_kernel = Label(window, text="Kernel")
SVM_kernel.grid(row=6, column=4, sticky=E)

SVM_kernel_var = StringVar()
SVM_kernel_entrybox = Entry(window, width=10, textvariable=SVM_kernel_var)
SVM_kernel_entrybox.grid(row=6, column=5)

SVM_R = Label(window, text="Cost")
SVM_R.grid(row=6, column=6, sticky=E)

SVM_R_var = IntVar()
SVM_R_entrybox = Entry(window, width=5, textvariable = SVM_R_var)
SVM_R_entrybox.grid(row=6, column=7)


def SVM():
    global DF, X, y, SVM_model
    
    # SVM
    Kernel = SVM_kernel_var.get()
    Cost = SVM_R_var.get()
    SVM_model= SVR(C=Cost, kernel=Kernel)
    SVM_model.fit(X_train, y_train)
    
    y_pred = SVM_model.predict(X_train)
    SVM_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    SVM_r2score_Train = round(r2_score(y_train, y_pred), 2)
    SVM_Tr, SVM_Tr_r2 = str(SVM_rmse_Train), str(SVM_r2score_Train)

    y_pred = SVM_model.predict(X_test)
    SVM_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    SVM_r2score_Test = round(r2_score(y_test, y_pred), 2)
    SVM_Te, SVM_Te_r2 = str(SVM_rmse_Test), str(SVM_r2score_Test)
  
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=6, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + SVM_Tr + ', ' + SVM_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=6, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + SVM_Te + ', ' + SVM_Te_r2))

SVM_Button = Button(window, text="RUN", command=SVM)
SVM_Button.grid(row=6, column=1)

#--------------------------------------------------------------------------------------------------------------------------------
# 3. Decision Tree

from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

DT = Label(window, text="Decision Tree")
DT.grid(row=7, column=0,sticky=E)

DT_Criteria = Label(window, text="Criteria") # {'friedman_mse', 'absolute_error', 'poisson', 'squared_error'}
DT_Criteria.grid(row=7, column=4, sticky=E)  

DT_Criteria_var = StringVar()
DT_Criteria_entrybox = Entry(window, width=10, textvariable=DT_Criteria_var)
DT_Criteria_entrybox.grid(row=7, column=5)

DT_Maxdept = Label(window, text="Max Depth")
DT_Maxdept.grid(row=7, column=6 ,sticky=E)

DT_Maxdept_var = IntVar()
DT_Maxdept_entrybox = Entry(window, width=5, textvariable=DT_Maxdept_var)
DT_Maxdept_entrybox.grid(row=7, column=7)


def DT():
    global DF, X, y, DT_model, DT_model, DT_rmse_Test, DT_rmse_Train, Criteria, Depth
    
    # DT 
    Criteria = DT_Criteria_var.get()
    Depth = DT_Maxdept_var.get()
    DT_model = DecisionTreeRegressor(criterion=Criteria, splitter='best', max_depth=Depth)
    print(DT_model)
    DT_model.fit(X_train, y_train)

    y_pred = DT_model.predict(X_train)
    DT_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    DT_r2score_Train = round(r2_score(y_train, y_pred), 2)
    DT_Tr, DT_Tr_r2 = str(DT_rmse_Train), str(DT_r2score_Train)

    y_pred = DT_model.predict(X_test)
    DT_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    DT_r2score_Test = round(r2_score(y_test, y_pred), 2)
    DT_Te, DT_Te_r2 = str(DT_rmse_Test), str(DT_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=7, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + DT_Tr + ', ' + DT_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=7, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + DT_Te + ', ' + DT_Te_r2))

DT_Button = Button(window, text="RUN", command=DT)
DT_Button.grid(row=7, column=1)

#--------------------------------------------------------------------------------------------------------------------------------
# 4. Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RF = Label(window, text="Random Forest")
RF.grid(row=8, column=0, sticky=E)

RF_Criteria = Label(window, text="Criteria")
RF_Criteria.grid(row=8, column=4, sticky=E)

RF_Criteria_var = StringVar()
RF_Criteria_entrybox = Entry(window, width=10, textvariable=RF_Criteria_var)
RF_Criteria_entrybox.grid(row=8, column=5)

RF_Maxdept = Label(window, text="Maximum Depth")
RF_Maxdept.grid(row=8, column=6, sticky=E)

RF_Maxdept_var = IntVar()
RF_Maxdept_entrybox = Entry(window, width=5, textvariable=RF_Maxdept_var)
RF_Maxdept_entrybox.grid(row=8, column=7)

RF_Estimator = Label(window, text="N_Estimators")
RF_Estimator.grid(row=8, column=8, sticky=E)

RF_Estimator_var = IntVar()
RF_Estimator_entrybox = Entry(window, width=5, textvariable=RF_Estimator_var)
RF_Estimator_entrybox.grid(row=8, column=9)


def RF():
    global DF, X,y, RF_model
    
    # RF
    
    Criteria = RF_Criteria_var.get()
    Depth = RF_Maxdept_var.get()
    Estimator = RF_Estimator_var.get()
    RF_model = RandomForestRegressor(n_estimators=Estimator,criterion=Criteria, max_depth=Depth)
    RF_model.fit(X_train, y_train)

    y_pred = RF_model.predict(X_train)
    RF_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    RF_r2score_Train = round(r2_score(y_train, y_pred), 2)
    RF_Tr, RF_Tr_r2 = str(RF_rmse_Train), str(RF_r2score_Train)

    y_pred = RF_model.predict(X_test)
    RF_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    RF_r2score_Test = round(r2_score(y_test, y_pred), 2)
    RF_Te, RF_Te_r2 = str(RF_rmse_Test), str(RF_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=8, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + RF_Tr + ', ' + RF_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=8, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + RF_Te + ', ' + RF_Te_r2))

RF_Button = Button(window, text="RUN", command=RF)
RF_Button.grid(row=8, column=1)

#--------------------------------------------------------------------------------------------------------------------------------

# 6. ADA Boosting

from sklearn.ensemble import AdaBoostRegressor

ADB = Label(window, text="ADA Boosting")
ADB.grid(row=9, column=0,sticky=E)

ADB_Criteria = Label(window, text="Estimator")
ADB_Criteria.grid(row=9, column=4, sticky=E)

ADB_Estimator_var = IntVar()
ADB_Estimator_entrybox = Entry(window, width=10, textvariable=ADB_Estimator_var)
ADB_Estimator_entrybox.grid(row=9, column=5)


def ADBoost():
    global DF,X,y, ADB_model
    
    N_Estimator=ADB_Estimator_var.get()
    ADB_model = AdaBoostRegressor(n_estimators=N_Estimator)
    ADB_model.fit(X_train, y_train)

    y_pred=ADB_model.predict(X_train)
    ADB_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    ADB_r2score_Train = round(r2_score(y_train, y_pred), 2)
    ADB_Tr, ADB_Tr_r2 = str(ADB_rmse_Train), str(ADB_r2score_Train)

    y_pred = ADB_model.predict(X_test)
    ADB_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    ADB_r2score_Test = round(r2_score(y_test, y_pred), 2)
    ADB_Te, ADB_Te_r2 = str(ADB_rmse_Test), str(ADB_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=9, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + ADB_Tr + ', ' + ADB_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=9, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + ADB_Te + ', ' + ADB_Te_r2))

ADB_Button = Button(window, text="RUN", command=ADBoost)
ADB_Button.grid(row=9, column=1)

#--------------------------------------------------------------------------------------------------------------------------------
# 7. Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

GB = Label(window, text='Gradient Boosting')
GB.grid(row=10, column=0, sticky=E)

GB_loss = Label(window, text="loss")
GB_loss.grid(row=10, column=4, sticky=E)

GB_loss_var = StringVar()
GB_loss_entrybox = Entry(window, width=10, textvariable=GB_loss_var)
GB_loss_entrybox.grid(row=10, column=5)

GB_Criteria = Label(window, text="n_estimators")
GB_Criteria.grid(row=10, column=6, sticky=E)

GB_Estimator_var = IntVar()
GB_Estimator_entrybox = Entry(window, width=5, textvariable=GB_Estimator_var)
GB_Estimator_entrybox.grid(row=10, column=7)

GB_lr = Label(window, text="learning_rate")
GB_lr.grid(row=10, column=8, sticky=E)

GB_lr_var = IntVar()
GB_lr_entrybox = Entry(window, width=5, textvariable=GB_lr_var)
GB_lr_entrybox.grid(row=10, column=9)

def GBoost():
    global DF,X,y, GB_model, loss, learning_rate
    
    N_Estimator = GB_Estimator_var.get()
    loss = GB_loss_var.get()
    learning_rate = GB_lr_var.get()

    GB_model = GradientBoostingRegressor(n_estimators=N_Estimator, loss=loss, learning_rate=learning_rate)
    GB_model.fit(X_train, y_train)

    y_pred=GB_model.predict(X_train)
    GB_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    GB_r2score_Train = round(r2_score(y_train, y_pred), 2)
    GB_Tr, GB_Tr_r2 = str(GB_rmse_Train), str(GB_r2score_Train)

    y_pred = GB_model.predict(X_test)
    GB_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    GB_r2score_Test = round(r2_score(y_test, y_pred), 2)
    GB_Te, GB_Te_r2 = str(GB_rmse_Test), str(GB_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=10, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + GB_Tr + ', ' + GB_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=10, column=4)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + GB_Te + ', ' + GB_Te_r2))

GB_Button = Button(window, text="RUN", command=GBoost)
GB_Button.grid(row=10, column=1)

#--------------------------------------------------------------------------------------------------------------------------------
# 7. XG Boost

import xgboost as xg  #pip install xgboost

XGB = Label(window, text='XG Boost')
XGB.grid(row=11, column=0, sticky=E)

XGB_Criteria = Label(window, text="n_estimators")
XGB_Criteria.grid(row=11, column=4, sticky=E)

XGB_Estimator_var = IntVar()
XGB_Estimator_entrybox = Entry(window, width=5, textvariable=XGB_Estimator_var)
XGB_Estimator_entrybox.grid(row=11, column=5)


def XGBoost():
    global DF,X,y, XGB_model, loss, learning_rate
    
    N_Estimator = XGB_Estimator_var.get()
    
    XGB_model = xg.XGBRegressor(n_estimators=N_Estimator)
    XGB_model.fit(X_train, y_train)

    y_pred = XGB_model.predict(X_train)
    XGB_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    XGB_r2score_Train = round(r2_score(y_train, y_pred), 2)
    XGB_Tr, XGB_Tr_r2 = str(XGB_rmse_Train), str(XGB_r2score_Train)

    y_pred = XGB_model.predict(X_test)
    XGB_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    XGB_r2score_Test = round(r2_score(y_test, y_pred), 2)
    XGB_Te, XGB_Te_r2 = str(XGB_rmse_Test), str(XGB_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=11, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + XGB_Tr + ', ' + XGB_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=11, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + XGB_Te + ', ' + XGB_Te_r2))

GB_Button = Button(window, text="RUN", command=XGBoost)
GB_Button.grid(row=11, column=1)

#--------------------------------------------------------------------------------------------------------------------------------
# 8. Stacking

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from vecstack import stacking  # pip install vecstack

STACK = Label(window, text="Stacking")
STACK.grid(row=12, column=0, sticky=E)

def Stacking():
    global DF, X, y, STACK_Train, STACK_Test, LR_model_stack
    
    models = [
              XGBRegressor(objective ='reg:linear', n_estimators = 5, seed = 42, n_jobs = -1),
              RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, max_depth=5),
              DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=5)
             ]
    STACK_Train, STACK_Test = stacking(models,                   
                           X_train, y_train, X_test,   
                           regression=TRUE,   
                           metric=mean_squared_error,           
                           shuffle=True,              
                           random_state=42,            
                           verbose=2)
    LR_model_stack = LinearRegression()
    LR_model_stack.fit(STACK_Train, y_train)
    
    y_pred = LR_model_stack.predict(STACK_Train)
    Stack_rmse_Train = round(mean_squared_error(y_train, y_pred), 2)
    Stack_r2score_Train = round(r2_score(y_train, y_pred), 2)
    Stack_Tr, Stack_Tr_r2 = str(Stack_rmse_Train), str(Stack_r2score_Train)

    y_pred = LR_model_stack.predict(STACK_Test)
    Stack_rmse_Test = round(mean_squared_error(y_test, y_pred), 2)
    Stack_r2score_Test = round(r2_score(y_test, y_pred), 2)
    Stack_Te, Stack_Te_r2 = str(Stack_rmse_Test), str(Stack_r2score_Test)
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=12, column=2)
    Confirm_entrybox.insert(1,str("Tr_RMSE, r2: " + Stack_Tr + ', ' + Stack_Tr_r2))
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=12, column=3)
    Confirm_entrybox.insert(1, str("Te_RMSE, r2: " + Stack_Te + ', ' + Stack_Te_r2))
    
STACK_Button = Button(window, text="RUN", command=Stacking)
STACK_Button.grid(row=12, column=1)

#-------------------------------------------------------------------------------------------------------------------------------
# 9. Model Pickling

# Creating an label for Subheader
Pickling = Label(window, text= " ---# Step 6: Pickling the Models #---")
Pickling.grid(row=13, column=0, sticky = W)

#---------------------------------------------------------
# Pickle the model

PIC = Label(window, text="Model Freeze")
PIC.grid(row=14, column=0, sticky=E)

def PICKLE():
    import pickle
    filename = 'Loinear_Regression.sav'
    pickle.dump(LR_model, open(filename, 'wb'))
       
    filename = 'SVR.sav'
    pickle.dump(SVM_model, open(filename, 'wb')) 
    
    filename = 'Decision_Tree_Regressor.sav'
    pickle.dump(DT_model, open(filename, 'wb'))
    
    filename = 'Random_Forest_Regressor.sav'
    pickle.dump(RF_model, open(filename, 'wb'))
    
    filename = 'ADA_Boost_Regressor.sav'
    pickle.dump(ADB_model, open(filename, 'wb'))

    filename = 'Gradient_Boost_Regressor.sav'
    pickle.dump(GB_model, open(filename, 'wb'))

    filename = 'XGBoost_Regressor.sav'
    pickle.dump(XGB_model, open(filename, 'wb'))
    
    filename = 'Stacking_Regressor.sav'
    pickle.dump(LR_model_stack, open(filename, 'wb'))
    
    Confirm_entrybox = Entry(window, width=10)
    Confirm_entrybox.grid(row=14, column=2)
    Confirm_entrybox.insert(1,str("Done"))
    
Pickle_Button = Button(window, text="Execute", command=PICKLE)
Pickle_Button.grid(row=14, column=1)

#-------------------------------------------------------------------------------------------------------------------------------
# 10. Machine Learning Operations

# Creating an label for Subheader
mlops = Label(window, text= " ---# Step 7: MLOps #---")
mlops.grid(row=15, column=0, sticky = W)

#---------------------------------------------------------
# Metrics for MLOps

# Import the libraries
import mlflow           # pip install mlflow if not installed
from datetime import datetime
import pickle
import numpy as np

# Create a label for mlflow parameters
metrics_criteria = Label(window, text="Model Loading and Metrics")
metrics_criteria.grid(row=16, column=0, sticky=E)

metrics_Estimator_var = StringVar()
metrics_Estimator_entrybox = Entry(window, width=30, textvariable=metrics_Estimator_var)
metrics_Estimator_entrybox.grid(row=16, column=1)

def get_metrics(y_true, y_pred):
    global mae, mse, msle, r2, adj_rand
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, adjusted_rand_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_rand = adjusted_rand_score(y_true, y_pred)
    return {'MAE': round(mae, 2), 'MSE': round(mse, 2), 'MSLE': round(msle, 2), 
            'r2_score': round(r2, 2), 'adj_rand_score': round(adj_rand, 2)}

def mlflow_metrics():
    global mae, mse, msle, r2, adj_rand, model, loaded_model, y_pred, run_metrics
    model = metrics_Estimator_var.get()
    loaded_model = pickle.load(open(model, 'rb'))
    y_pred = loaded_model.predict(X_test)
    run_metrics = str(get_metrics(y_test, y_pred))

    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=16, column=3)
    Confirm_entrybox.insert(1,str({'Metrics': run_metrics}))
    
metrics_Button = Button(window, text="RUN", command=mlflow_metrics)
metrics_Button.grid(row=16, column=2)

#---------------------------------------------------------
# MLOps - Registration and Transition

'''
### MLFlow model registry
- Run the below command from the terminal first before creating experiment and registering it
- `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5001`
- Set the MLflow tracking uri (within code section).
    mlflow.set_tracking_uri("http://localhost:5001")  
'''
# Creating and mlflow experiment and registering the model for the experiment
def create_exp_and_register_model(experiment_name, run_name, run_metrics, model, run_params=None):
    mlflow.set_tracking_uri("http://localhost:5001") 
    #use above line if you want to use any database like sqlite as backend storage for model else comment this line
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
              
        mlflow.set_tag("tag1", "Regressor")
        mlflow.set_tags({"tag2":"Retail_dataset_regression", "tag3":"Production"})
        mlflow.sklearn.log_model(model, "model", registered_model_name="retail-regressor")

# Create a label for mlflow parameters
mlflow_criteria = Label(window, text='MLFLOW - Registration and Transition')
mlflow_criteria.grid(row=17, column=0, sticky=E)


def mlflow_exp():
    global experiment_name, run_name, run_metrics

    # Naming the experiments for MLflow Tuned model
    experiment_name = "Retail_Regressor" +str(datetime.now().strftime("%d-%m-%y"))
    run_name="Retail_regression_model" +str(datetime.now().strftime("%d-%m-%y"))
    run_metrics = get_metrics(y_test, y_pred)
    create_exp_and_register_model(experiment_name, run_name, run_metrics, loaded_model)

    # Model Transition
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(  name="retail-regressor",
                                            version=1,
                                            stage="Production")
    
    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=17, column=2)
    Confirm_entrybox.insert(1,str('Experiment created and Transitioned'))

    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=17, column=3)
    Confirm_entrybox.insert(1,str('Track the experiment at http://localhost:5001'))

mlops_Button = Button(window, text="RUN", command=mlflow_exp)
mlops_Button.grid(row=17, column=1)

#---------------------------------------------------------
# MLOps - Model Serving and Prediction

'''
## Add the below command to the environmental variable, 
    - Go the edit environmental variable
    - click on new and then give the variable name as MLFLOW_TRACKING_URI and value as - http://localhost:5001'
## **Now run this command from command line**
make sure we use to write the different port - other than the one you used while starting mlflow server
    `mlflow models serve --model-uri models:/retail-regressor/Production -p 6002 --env-manager=local`

'''

import requests
import pandas as pd

# Create a label for mlflow parameters
mlflow_criteria1 = Label(window, text='MLFLOW - Model Serving')
mlflow_criteria1.grid(row=18, column=0, sticky=E)

mlflow_Estimator_var1 = StringVar()
mlflow_Estimator_entrybox1 = Entry(window, width=30, textvariable=mlflow_Estimator_var1)
mlflow_Estimator_entrybox1.grid(row=18, column=1)

def model_serving():
    global test_data, response

    test_data = mlflow_Estimator_var1.get()
    df = pd.read_csv(test_data)
    df.head()
    lst = df.values.tolist()
    inference_request = {
                        "data": lst
                        }
    endpoint = "http://localhost:6002/invocations"
    response = requests.post(endpoint, json=inference_request)
    result = response.text
    print(response.text)

    Confirm_entrybox = Entry(window, width=15)
    Confirm_entrybox.grid(row=18, column=3)
    Confirm_entrybox.insert(1,str('Endpoint : '))

    Confirm_entrybox = Entry(window, width=30)
    Confirm_entrybox.grid(row=18, column=4)
    Confirm_entrybox.insert(1,str('http://localhost:6002/invocations'))

    #Confirm_entrybox = Entry(window, width=30)
    #Confirm_entrybox.grid(row=18, column=4)
    #Confirm_entrybox.insert(1,str('Predicted_Result:', str(result[0])))

mlops_Button1 = Button(window, text="RUN", command=model_serving)
mlops_Button1.grid(row=18, column=2)

window.mainloop()