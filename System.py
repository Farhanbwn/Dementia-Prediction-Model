import pickle
import numpy as np

### Loading the Saved Model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (1,0,0,0,87,14,2.0,27.0,0.0,1987,0.696,0.883)

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