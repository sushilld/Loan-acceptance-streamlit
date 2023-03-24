import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Loan Data Visualization and Prediction", layout="wide")
st.title('Loan Data Visualization and Prediction')
tab1, tab2 = st.tabs(['Visualization', 'Prediction'])

df = pd.read_csv('loan_train.csv')
df['Gender'].fillna(df['Gender'].mode()[0],inplace = True)
df['Married'].fillna(df['Married'].mode()[0],inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace = True)
df['Term'].fillna(df['Term'].mode()[0],inplace = True)

df['Loan_Amount'].fillna(df['Loan_Amount'].mean(),inplace = True) 

with tab1:
    st.success('## This is Visualization Section')
    st.write('Loan accepted vs rejected') 
    nsamp = df['Status'].value_counts().reset_index()
    st.bar_chart(nsamp)
    st.write('Applicant Income')     
    st.line_chart(df['Applicant_Income'])
    st.write('Male vs Female Applicants') 
    nsamp = df['Gender'].value_counts().reset_index()
    st.bar_chart(nsamp)
    st.write('Co-applicant Income')     
    st.line_chart(df['Applicant_Income'])
    st.write('Self Employed Applicants') 
    nsamp = df['Self_Employed'].value_counts().reset_index()
    st.bar_chart(nsamp)
    st.write('Credit History') 
    nsamp = df['Credit_History'].value_counts().reset_index()
    st.bar_chart(nsamp)

with tab2:
    model_svm = joblib.load('model.pkl')
    st.success('## This is Prediction Section')
    Gender= st.selectbox('Gender',('Male','Female'))
    Married= st.selectbox('Married',('No','Yes'))
    Dependents= st.selectbox('Number Of Dependents',('0','1','2','3 or More Dependents'))
    Education= st.selectbox('Education status',('Graduate','Not Graduate'))
    Self_Employed= st.selectbox('Self Employed',('No','Yes'))
    ApplicantIncome= st.number_input('Applicant Income',0)
    CoapplicantIncome= st.number_input('Coapplicant Income',0)
    LoanAmount= st.number_input('Loan Amount',50000)
    Loan_Amount_Term= st.select_slider('Loan Amount Term i Days',range(180, 360*5, 180))
    Credit_History= st.select_slider('Credit History 1 for Good 0 for Bad',[0,1])
    Property_Area= st.selectbox('Area of Property',('Urban','Rural','Semiurban'))

    columns= ['Gender','Married','Dependents','Education','Self_Employed','Applicant_Income','Coapplicant_Income',
            'Loan_Amount','Term','Credit_History','Area']

    def predict():
        df.drop('Status', axis=1, inplace=True)
        col= np.array([Gender,Married,Dependents,Education,Self_Employed,
                    int (ApplicantIncome),float(CoapplicantIncome),int(LoanAmount),float(Loan_Amount_Term),float(Credit_History),Property_Area])
        data= pd.DataFrame([col],columns=columns)

        # df_das = df.append(data, ignore_index = True)
        df_das = pd.concat([df, data], ignore_index=True)
        print(df_das.info())     
        df_das['Applicant_Income'] = df_das['Applicant_Income'].astype(np.int64)
        df_das['Loan_Amount'] = df_das['Loan_Amount'].astype(np.int64)
        df_das['Coapplicant_Income'] = df_das['Coapplicant_Income'].astype(np.float64)
        df_das['Term'] = df_das['Term'].astype(np.float64)
        df_das['Credit_History'] = df_das['Credit_History'].astype(np.float64)
        
        print(df_das.info())

        df_das = pd.get_dummies(df_das)
        print('-*'*35)
        print(df_das)

        # Drop columns
        df_das = df_das.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
                    'Self_Employed_No'], axis = 1)

        # Rename columns name
        new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
            'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
            'Loan_Status_Y': 'Loan_Status'}
            
        df_das.rename(columns = new, inplace = True)

        data = df_das.tail(1).to_numpy()
        scaler_model = joblib.load('scaler.pkl')
        data = scaler_model.transform(data)
        print(data)        
        prediction= model_svm.predict(data)[0]
        print(prediction)

        if prediction == 1:
            st.success('You Can Get The Loan:thumbsup:')
            st.balloons()
        else:
            st.error('Sorry You Cant Get The Loan:thumbsdown:')
            st.snow()


    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0099ff;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #00ff00;
        color:#ff0000;
        }
    </style>""", unsafe_allow_html=True)


    st.button('Predict',on_click=predict)