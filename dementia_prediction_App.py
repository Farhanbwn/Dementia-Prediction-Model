import pickle
import numpy as np
import streamlit as st
from pymongo import MongoClient
import datetime 



# Connect to MongoDB Atlas
client = MongoClient(st.secrets["mongodb"]["uri"])
db = client["DementiaDB"]
collection = db["Dataset"]



### Loading the Saved Model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))



def dementia_prediction(input_data):

    ### changing the list data into numpy arrays
    input_data_as_numpy = np.asarray(input_data)

    # this reshape tell the model we need the prediction for only one data
    input_data_reshape = input_data_as_numpy.reshape(1,-1)

    # printing the prediction 
    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    # printing the prediction 
    if(prediction[0] == '0'):
        return "The patient is NonDemented" 
    elif(prediction[0] == '1'):
        return "The patient is Demented" 
    else:
        return "The patient is Converted" 
    

def main():

   
    ###giving title of Web App
    st.title('Dementia Prediction Web App')

    ### taking the input in the variables from the user
    Name = st.text_input('Enter the Patient Name')
    # Row 1
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        Visit = st.text_input('Visit')
    with r1c2:
        MR_Delay = st.text_input('MR Delay')
    with r1c3:
        Gender = st.text_input('Gender (M/F)')
    with r1c4:
        Hand = st.text_input('Hand (R/L)')

    # Row 2
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        Age = st.text_input('Age')
    with r2c2:
        EDUC = st.text_input('Education (Years)')
    with r2c3:
        SES = st.text_input('Socioeconomic Status')
    with r2c4:
        MMSE = st.text_input('MMSE Score')

    # Row 3
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        CDR = st.text_input('CDR')
    with r3c2:
        eTIV = st.text_input('eTIV')
    with r3c3:
        nWBV = st.text_input('nWBV')
    with r3c4:
        ASF = st.text_input('ASF')


    diagnosis = ''

    button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns(5)
    with button_col3:
        if st.button("Test Result"):
            
            # Check for any empty fields
            required_fields = [Name, Visit, MR_Delay, Gender, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF]
            if any(field.strip() == '' for field in required_fields):
                    st.error("Please fill in all input fields.")
                    return

            # Convert Gender input
            if Gender.lower() == 'M' or Gender.lower() == 'm':
                Gender_val = 0
            elif Gender.lower() == 'F' or Gender.lower() == 'f':
                Gender_val = 1
            else:
                st.error("Please enter 'M' or 'F' for Gender.")
                return

            # Convert Hand input
            if Hand.lower() == 'R' or Hand.lower() == 'r':
                Hand_val = 0
            elif Hand.lower() == 'L' or Hand.lower() == 'L':
                Hand_val = 1
            else:
                st.error("Please enter 'R' or 'L' for Hand.")
                return
    
            diagnosis = dementia_prediction([Visit,MR_Delay,Gender_val,Hand_val,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF])


            try:
                input_values = [
                    float(Visit), float(MR_Delay), Gender_val, Hand_val,
                    float(Age), float(EDUC), float(SES), float(MMSE),
                    float(CDR), float(eTIV), float(nWBV), float(ASF)
                ]
            except ValueError:
                st.error("Please enter numeric values in all fields except Gender and Hand.")
                return

            diagnosis = dementia_prediction(input_values)

            # Save to MongoDB Atlas
            record = {
                "timestamp": datetime.datetime.now(),
                "Name": Name,
                "Visit": Visit,
                "MR_Delay": MR_Delay,
                "Gender": Gender,
                "Hand": Hand,
                "Age": Age,
                "EDUC": EDUC,
                "SES": SES,
                "MMSE": MMSE,
                "CDR": CDR,
                "eTIV": eTIV,
                "nWBV": nWBV,
                "ASF": ASF,
                "Prediction": diagnosis
            }

            collection.insert_one(record)
            # st.success("Prediction saved to database.")

         
    # Show diagnosis in center
    if diagnosis:
        st.markdown(
            f"<h3 style='text-align: center; color: green;'>{diagnosis}</h3>",
            unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()