### Import all the libary 

import numpy as np  # numpy is used for numpy arrays
import pandas as pd # pandas is used for creating the data table in a structural way
from sklearn.preprocessing import StandardScaler # Standaed scaler is used for standardize the data 
from sklearn.model_selection import train_test_split #train_test_split is used for spliting the data into train and test parts
from sklearn import svm #Support vector machine (SVM) is the classifer of the project
from sklearn.metrics import accuracy_score # accuracy_score is used for to see the accuracy score of the model 
from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns


### read the dataset to pandas dataframe
dataset = pd.read_csv("dementia_dataset.csv")


### printing the first 5 rows of the dataset
# print(dataset.head())


### printing to check if there was a null value
# print(dataset.isnull())

### printing coloumwise total number of null values
# print(dataset.isnull().sum())

### printing total number of null values
# print(dataset.isnull().sum().sum())

### ### Replaceing Missing values to Median of the Column
simple_median = SimpleImputer(strategy='median')
dataset['SES'] = simple_median.fit_transform(dataset[['SES']])
dataset['MMSE'] = simple_median.fit_transform(dataset[['MMSE']])


### Rechacking the missing values
# print(dataset.isnull().sum())


### printing number of rows and columns
# print(dataset.shape)


### printong the statistical values of the data 
# print(dataset.describe())



### printing the total of Demeneted, Nondemented and Converted pesent
# print(dataset['Group'].value_counts())


### Replacing data values
## Nondemented--->> 0
# ## Demented--->> 1
# ## Converted--->> 2


dataset = dataset.replace(to_replace='Nondemented',value= '0')
dataset = dataset.replace(to_replace='Demented',value= '1')
dataset = dataset.replace(to_replace='Converted',value= '2')


# ## Male--->> 0
# ## Female--->> 1
# pd.set_option('future.no_silent_downcasting', True)
dataset = dataset.replace(to_replace='M',value= '0')
dataset = dataset.replace(to_replace='F',value= '1')


# ## Right hand--->> 0
# ## Left hand--->> 1

dataset = dataset.replace(to_replace='R',value= '0')
dataset = dataset.replace(to_replace='L',value= '1')




# ## printing the mean value of pesent
# # print(dataset.groupby('Group').mean())

# ### checking missing values again
# # print(dataset.isnull().sum())



### separte the Group columns to the main dataset
y = dataset['Group']


### drop 'Subject ID','MRI ID' from the main datset
x = dataset.drop(columns=['Subject ID','MRI ID','Group'], axis=1)

### convert all data String into int
x = x.astype(int)

# print(x)
# print(y)

# # dataset.to_csv("dementia_new_dataset.csv")


### Data Standardization 
Stand = StandardScaler()
x = x.values
Stand.fit(x)
Stand_data = Stand.transform(x)

# # # print(Stand_data)


### spliting the standardize data into train and test data
x = Stand_data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
### test_size=0.2---> represent 80% of train data and 20% of test data
### tratify=y---> represent to similar proportional of splitting the data 
### random_state=2---> random splitting the data

# printing the number of train and test data
# print(x.shape,x_train.shape,x_test.shape)



### import the dataset into the model(SVM)
### training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)



# # # # # ### chaecking the accuracy of the model of train data
x_train_predit = classifier.predict(x_train)
training_data_acc = accuracy_score(x_train_predit,y_train)

print("Accuracy score of x_train is ",training_data_acc)


# # # # ### chaecking the accuracy of the model of test data
x_test_predit = classifier.predict(x_test)
testing_data_acc = accuracy_score(x_test_predit,y_test)

print("Accuracy score of x_test is ",testing_data_acc)


# ### taking the input in the variables from the user
# # Visit = int(input("Enter the number of time the pesent visited :"))
# # MR_Delay = int(input("Enter the MR Delay :"))
# # Gender = (input("Enter the Gender of the pesent :"))
# # if(Gender == "M"):
# #     Gender = 0
# # elif(Gender == 'F'):
# #     Gender = 1
# # else:
# #     print("Please Enter M or F")
# # Hand = (input("Enter the Hand of the pesent :"))
# # if(Hand == "R"):
# #     Hand = 0
# # elif(Hand == 'L'):
# #     Hand = 1
# # else:
# #     print("Please Enter R or L")
# # Age = int(input("Enter the Age of the pesent:"))
# # EDUC = int(input("Enter the EDUC of the pesent :"))
# # SES = int(input("Enter the SES of the pesent :"))
# # MMSE = int(input("Enter the MMSR:"))
# # CDR = int(input("Enter the CDR:"))
# # eTIV = int(input("Enter the eTIV:"))
# # nWBV = float(input("Enter the nWBV:"))
# # ASF = float(input("Enter the ASF:"))


# # # ### taking the input from the user
# # # input_data = (Visit,MR_Delay,Gender,Hand,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF)

input_data = (1,0,0,0,72,20,1.0,26.0,0.5,1911,0.719,0.919)

### changing the list data into numpy arrays
input_data_as_numpy = np.asarray(input_data)


# this reshape tell the model we need the prediction for only one data
input_data_reshape = input_data_as_numpy.reshape(1,-1)

# standardize the data to get the output
# std_data = Stand.transform(input_data_reshape)
# print(std_data)

# printing the prediction 
prediction = classifier.predict(input_data_reshape)
print(prediction)

# printing the prediction 
if(prediction[0] == '0'):
    print("The patient is NonDemented")
elif(prediction[0] == '1'):
    print("The patient is Demented")
else:
    print("The patient is Converted")


### Saving the Trained Model
import pickle

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

### Loading the Saved Model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (1,0,0,0,72,20,1.0,26.0,0.5,1911,0.719,0.919)

### changing the list data into numpy arrays
input_data_as_numpy = np.asarray(input_data)


# this reshape tell the model we need the prediction for only one data
input_data_reshape = input_data_as_numpy.reshape(1,-1)

# standardize the data to get the output
# std_data = Stand.transform(input_data_reshape)
# print(std_data)

# printing the prediction 
prediction = loaded_model.predict(input_data_reshape)
print(prediction)

# printing the prediction 
if(prediction[0] == '0'):
    print("The patient is NonDemented")
elif(prediction[0] == '1'):
    print("The patient is Demented")
else:
    print("The patient is Converted")

