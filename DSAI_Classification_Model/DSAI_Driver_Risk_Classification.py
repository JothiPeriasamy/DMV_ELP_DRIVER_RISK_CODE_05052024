"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
© Copyright 2022, California, Department of Motor Vehicle, all rights reserved.
The source code and all its associated artifacts belong to the California Department of Motor Vehicle (CA, DMV), and no one has any ownership
and control over this source code and its belongings. Any attempt to copy the source code or repurpose the source code and lead to criminal
prosecution. Don't hesitate to contact DMV for further information on this copyright statement.

Release Notes and Development Platform:
The source code was developed on the Google Cloud platform using Google Cloud Functions serverless computing architecture. The Cloud
Functions gen 2 version automatically deploys the cloud function on Google Cloud Run as a service under the same name as the Cloud
Functions. The initial version of this code was created to quickly demonstrate the role of MLOps in the ELP process and to create an MVP. Later,
this code will be optimized, and Python OOP concepts will be introduced to increase the code reusability and efficiency.
____________________________________________________________________________________________________________
Development Platform                | Developer       | Reviewer   | Release  | Version  | Date
____________________________________|_________________|____________|__________|__________|__________________
Google Cloud Serverless Computing   | DMV Consultant  | Ajay Gupta | Initial  | 1.0      | 09/18/2022

-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

import streamlit as vAR_st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import shap
from IPython.display import display, HTML
import random
from tempfile import NamedTemporaryFile
# from streamlit_chunk_file_uploader import uploader
import traceback

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lime import lime_tabular
import streamlit.components.v1 as components
import base64
import sweetviz
from DSAI_Bigquery_Impl.DSAI_GCP_Operations import RequestToBigquery,TestDataResponseToBigquery,Upload_Data_To_GCS

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer

from DSAI_GENAI.DSAI_Azure_Assistant import DataInsightsWithAssistant
import os

AZURE_ASSISTANT_ID = os.environ["AZURE_ASSISTANT_ID"]



def DriverRiskClassification():
    
    try:
        if 'training_data' not in vAR_st.session_state:
            vAR_st.session_state.training_data = None
        
        if 'vAR_model' not in vAR_st.session_state:
            vAR_st.session_state.vAR_model = None
            
        if 'vAR_tested_log' not in vAR_st.session_state:
            vAR_st.session_state.vAR_tested_log = None

        
        vAR_df = None
        vAR_test = None
        vAR_test_dataset = None

        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        vAR_result_data = pd.DataFrame()
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Upload Training Data')

        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            # vAR_dataset = vAR_st.button("Read Bigquery Table")
            vAR_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="reg")
            
            if vAR_dataset:

                # with NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:  # Create temporary file
                #     temp_file.write(vAR_dataset.read())


                vAR_raw_df = pd.read_csv(vAR_dataset)

                print('len - ',len(vAR_raw_df))





                if vAR_st.session_state.training_data is None:
                    # RequestToBigquery(vAR_raw_df)
                    Upload_Data_To_GCS(vAR_raw_df,"request")
                    # This is to remove last 4 columns - CREATED_BY,CREATED_AT,UPDATED_BY,UPDATED_AT
                    vAR_st.session_state["training_data"] = vAR_raw_df.head(5000)
                    
        
                
        if vAR_st.session_state["training_data"] is not None:
            
            #Explore with GENAI
            
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.subheader('Analyze Data With GEN-AI')


            with col4:
                vAR_st.write('')
                vAR_st.write('')
                vAR_gen_data = vAR_st.radio("Response time depends on the request file size",["No","Yes"],horizontal=False)



            if vAR_gen_data=="Yes":
                
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
                with col4:
                    
                    vAR_user_input = vAR_st.text_area('Sample : Which court number has investigated more number of crashes?')

                    vAR_submit = vAR_st.button("Submit")
                col1,col2,col3 = vAR_st.columns([1.5,10,1.5])

                if vAR_submit:   
                    if vAR_user_input:
                        vAR_llm_response = DataInsightsWithAssistant(vAR_user_input,AZURE_ASSISTANT_ID)

                        with col2:
                            vAR_st.info(vAR_llm_response) 
                
                

            Preview_Data(vAR_st.session_state["training_data"],"train_preview")

            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])

            vAR_st.write('')
            vAR_st.write('')

            with col4:
                vAR_st.write('')
                vAR_st.write('')
                vAR_eda = vAR_st.button("Exploratory Data Analysis")


            if vAR_eda:
                col1,col2,col3 = vAR_st.columns([0.4,10,1])
                with col2:
                    vAR_analysis = sweetviz.analyze(vAR_st.session_state["training_data"],pairwise_analysis="on")


                    # vAR_analysis.show_html(filepath=r'C:\Users\Admin\Desktop\DMV_Driver_Risk_Analysis\DataAnalysis.html', open_browser=False, layout='vertical', scale=1.0)
                    vAR_analysis.show_html(filepath='/tmp/DataAnalysis.html', open_browser=False, layout='vertical', scale=1.0)

                    # with open(r'C:\Users\Admin\Desktop\DMV_Driver_Risk_Analysis\DataAnalysis.html', 'r') as f:
                    with open('/tmp/DataAnalysis.html', 'r') as f:
                        raw_html = f.read().encode("utf-8")
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                with col4:
                    vAR_st.download_button(label="Download EDA Report",
                                data=raw_html,
                                file_name="DriverRiskAnalysis.html",mime="text/html")
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.write('')
                    # components.iframe(src=src, width=1100, height=2500, scrolling=True)
            if vAR_st.session_state["training_data"] is not None:

                selector,vAR_features = Feature_Selection(vAR_st.session_state["training_data"])

                # if "selector" not in vAR_st.session_state:
                #     vAR_st.session_state["selector"] = selector

                col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
                with col2:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.info('**Note : We took only potential features based on EDA&RFE(Recursive Feature Elimination) Technnique.**')


            # Model Training
            if vAR_st.session_state["training_data"] is not None:

                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])

                with col2:

                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.subheader('Model Training')

                with col4:        
                    vAR_st.write('')
                    vAR_st.write('')

                    vAR_model_train = vAR_st.button("Train the Model")
                    vAR_st.write('')

                    vAR_st.write('')


                if vAR_model_train:
                    if "X_train" not in vAR_st.session_state and "X_train_cols" not in vAR_st.session_state:
                        vAR_st.session_state['vAR_model'],vAR_st.session_state['X_train'],vAR_st.session_state['X_train_cols'] = Train_Model(vAR_st.session_state["training_data"],vAR_features,selector)

                # Model Testing

                if vAR_st.session_state['vAR_model'] is not None:
                    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                    with col2:
                        vAR_st.write('')
                        vAR_st.write('')
                        vAR_st.subheader('Upload Test Data')

                    with col4:
                        vAR_st.write('')
                        vAR_test_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="test")

                if vAR_test_dataset is not None:   
                    vAR_test_data = pd.read_csv(vAR_test_dataset)

                    # Preview Test Data

                    Preview_Data(vAR_test_data,"test_preview")


                    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                    with col2:
                        vAR_st.write('')
                        vAR_st.write('')
                        vAR_st.subheader('Test Model')

                    with col4:
                        vAR_st.write('')
                        vAR_st.write('')
                        vAR_test = vAR_st.button("Test Model")



                if vAR_test:

                    vAR_test_data = Test_Model(vAR_test_data)

                    if not vAR_st.session_state.vAR_tested_log:
                        vAR_st.session_state["vAR_tested_log"] = True


                if vAR_st.session_state["vAR_tested_log"]:

                    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])

                    with col4:
                        if vAR_st.session_state["vAR_test_data"] is not None:
                            vAR_st.markdown(create_download_button(vAR_test_data), unsafe_allow_html=True)

                            vAR_st.write('')
                            vAR_st.write('')

                    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])

                    with col2:
                        vAR_st.write('')
                        vAR_st.write('')
                        vAR_st.subheader('Model Outcome')

                    with col4:
                        vAR_st.write('')
                        vAR_model_outcome_graph = vAR_st.button("Model Outcome Summary")

                    if vAR_model_outcome_graph:

                        ModelOutcomeSummary(vAR_st.session_state["vAR_test_data"])


                    # Uncomment below code for XAI

                    # col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])

                    # with col2:
                    #     vAR_st.write('')
                    #     vAR_st.write('')
                    #     vAR_st.write('')
                    #     vAR_st.subheader("Select Driver ID For XAI")



                    # with col4:
                    #     vAR_st.write('')
                    #     vAR_st.write('')

                    #     vAR_idx_values = ['Select Driver Id'] 
                    #     vAR_idx_values.extend([item for item in vAR_test_data.index])               
                    #     vAR_test_id = vAR_st.selectbox(' ',vAR_idx_values)



                    # if vAR_test_id!='Select Driver Id': 


                    #     vAR_df_columns = vAR_test_data.columns

                    #     vAR_numeric_columns = vAR_test_data._get_numeric_data().columns 

                    #     vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))

                    #     # Encode Test Data
                    #     # To fill categorical NaN column
                    #     for col in vAR_categorical_column:
                    #         vAR_test_data[col].fillna('Missing', inplace=True)

                    #     # To fill Integer NaN column
                    #     for col in vAR_numeric_columns:
                    #         vAR_test_data[col].fillna(10000, inplace=True)

                    #     # data_encoded = pd.get_dummies(vAR_train_df, columns=vAR_categorical_column)
                    #     encoder = OrdinalEncoder()
                    #     vAR_test_data[vAR_categorical_column] = encoder.fit_transform(vAR_test_data[vAR_categorical_column])

                    #     data_encoded = vAR_test_data
                    #     vAR_data_encoded_cols = data_encoded.columns

                    #     print('type dataencoded - ',type(data_encoded))
                    #     print('dataencoded cols - ',vAR_data_encoded_cols)

                    #     data_encoded = data_encoded[vAR_st.session_state['X_train_cols']]


                    #     vAR_model,X_train = vAR_st.session_state['vAR_model'],vAR_st.session_state['X_train']
                    #     features = X_train.columns

                    #     print('features - ',features)
                    #     col1,col2,col3 = vAR_st.columns([1,15,1])

                    #     with col2:

                    #         vAR_st.write('')
                    #         vAR_st.markdown('<hr style="border:2px solid gray;">', unsafe_allow_html=True)
                    #         vAR_st.write('')
                    #         vAR_st.markdown("<div style='text-align: center; color: black;font-weight:bold;'>Explainable AI with LIME(Local  Interpretable Model-agnostic Explanations) Technique</div>", unsafe_allow_html=True)
                    #         vAR_st.write('')
                    #         vAR_st.write('')
                    #         vAR_st.write('')
                    #         vAR_st.write('')
                    #         ExplainableAI(X_train,features,vAR_model,data_encoded,vAR_test_id)
                    #         # SHAPExplainableAI(X_train,vAR_model,data_encoded,vAR_test_id)





                    
            
            
    except BaseException as e:
        print('Error in Classification- ',str(e))
        print("Error Traceback - ",str(traceback.print_exc()))
                    
                    
                    
                
                
            
            
        
    
            
                
            
def Preview_Data(vAR_df,mode):
    vAR_df = vAR_df.head(50)
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Preview Data')
        

    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_preview_data = vAR_st.button("Preview Data",key=mode)
        
    
    
    if vAR_preview_data:
        col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
        with col2:
            
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.dataframe(data=vAR_df)
            
            
            

def Feature_Selection(vAR_df):
    
    vAR_columns =["All"]
    # vAR_potential_features = ["DRIVER_AGE","SEC1","COURT","DISM_CORR_IND","RES_COUNTY","YEARS_OF_EXP",
    #                           "DACTYPE","CRASH_TIME","NO_OF_INJURIES","NO_OF_FATALS","SOBRIETY","PHYS_COND","CITED"]
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    
    
    
    vAR_train_df = vAR_df.drop(vAR_df.columns[-1],axis=1)
    
    vAR_df_columns = vAR_train_df.columns
    
    print('vAR_df_columns - ',vAR_df_columns)
        
    vAR_numeric_columns = vAR_train_df._get_numeric_data().columns 
    
    vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
    print('vAR_df_cat_columns - ',vAR_categorical_column)
    
    # To fill categorical NaN column
    for col in vAR_categorical_column:
        vAR_train_df[col].fillna('Missing', inplace=True)
    
    # To fill Integer NaN column
    for col in vAR_numeric_columns:
        vAR_train_df[col].fillna(10000, inplace=True)
        
    print('traindf - ',vAR_train_df.head())
    
    # data_encoded = pd.get_dummies(vAR_train_df, columns=vAR_categorical_column)
    encoder = OrdinalEncoder()
    vAR_train_df[vAR_categorical_column] = encoder.fit_transform(vAR_train_df[vAR_categorical_column])
    
    data_encoded = vAR_train_df
    print('traindf encoded - ',data_encoded.head())
    vAR_data_encoded_cols = data_encoded.columns

    # Add the one-hot encoded variables to the dataset and remove the original 'Vehicle_Type' column
    # data_encoded.drop(vAR_categorical_column, axis=1, inplace=True)

    # Label encoding for 'Crash_Level'
    label_enc = LabelEncoder()
    data_encoded['Crash_Level'] = label_enc.fit_transform(data_encoded[data_encoded.columns[-1]])

    # Split the data into features (X) and target (y)
    X = data_encoded.drop(data_encoded.columns[-1],axis=1)
    y = vAR_df.iloc[: , -1:]
    
    vAR_columns.extend(vAR_df_columns)
    
    with col2:
        
        vAR_st.write('')
        vAR_st.subheader('Feature Selection')
        
    with col4:
        vAR_features = vAR_st.multiselect(' ',vAR_columns,default="All")
        vAR_st.write('')
        
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        vAR_st.write('')
        with vAR_st.expander("List selected features"):  
            if 'All' in vAR_features:
                vAR_st.write('Features:',vAR_columns[1:])
            else:
                for i in range(0,len(vAR_features)):
                    vAR_st.write('Feature',i+1,':',vAR_features[i])
                    
            
    
    print('features - ',vAR_features)   
    
    estimator = LogisticRegression(max_iter=10000)
    
    
    if 'All' in vAR_features:
        print('vAR_features in All scenario - ',vAR_df_columns)
        selector = RFE(estimator, n_features_to_select=len(vAR_df_columns))
        selector = selector.fit(X, y)
        return selector,vAR_df_columns
    
    else:
        selector = RFE(estimator, n_features_to_select=len(vAR_features))
        selector = selector.fit(X, y)
    
    
    
    print('top_features_rfe - ',selector.support_)
    
    # top_features = [i for i, x in enumerate(selector.support_) if x]

    
    

    return selector,vAR_features
                      
        
            
def Train_Model(vAR_df,vAR_features,selector):
    
    vAR_train_df = vAR_df.drop(vAR_df.columns[-1],axis=1)
    
    
    
    
    vAR_df_columns = vAR_train_df.columns
    
    print('vAR_df_columns - ',vAR_df_columns)
        
    vAR_numeric_columns = vAR_train_df._get_numeric_data().columns 
    
    vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
    print('vAR_df_cat_columns - ',vAR_categorical_column)
    
    # To fill categorical NaN column
    for col in vAR_categorical_column:
        vAR_train_df[col].fillna('Missing', inplace=True)
    
    # To fill Integer NaN column
    for col in vAR_numeric_columns:
        vAR_train_df[col].fillna(10000, inplace=True)
        
    print('traindf - ',vAR_train_df.head())
    
    # data_encoded = pd.get_dummies(vAR_train_df, columns=vAR_categorical_column)
    encoder = OrdinalEncoder()
    vAR_train_df[vAR_categorical_column] = encoder.fit_transform(vAR_train_df[vAR_categorical_column])
    
    data_encoded = vAR_train_df
    print('traindf encoded - ',data_encoded.head())
    vAR_data_encoded_cols = data_encoded.columns

    # Add the one-hot encoded variables to the dataset and remove the original 'Vehicle_Type' column
    # data_encoded.drop(vAR_categorical_column, axis=1, inplace=True)

    # Label encoding for 'Crash_Level'
    label_enc = LabelEncoder()
    data_encoded['Crash_Level'] = label_enc.fit_transform(data_encoded[data_encoded.columns[-1]])

    # Split the data into features (X) and target (y)
    X = data_encoded.drop(data_encoded.columns[-1],axis=1)
    X_train_raw = data_encoded.copy()
    X_train_raw.drop(X_train_raw.columns[-1],axis=1,inplace=True)
    
    print('X_train_raw cols - ',X_train_raw.columns)
    y = vAR_df.iloc[: , -1:]
    
    print('top_features in model train part - ',selector.support_)
    
    X = selector.transform(X)
    # vAR_df = vAR_df[["DRIVER_AGE","SEC1","COURT","DISM_CORR_IND","RES_COUNTY","YEARS_OF_EXP",
    #                           "DACTYPE","CRASH_TIME","NO_OF_INJURIES","NO_OF_FATALS","SOBRIETY","PHYS_COND","CITED","LABEL"]].copy()
    
    
    
    print('vAR_numeric_columns - ',vAR_numeric_columns)
    
    print('vAR_categorical_column - ',vAR_categorical_column)
    
    print('data_encoded cols - ',data_encoded.columns)
    
    
    
    
    
    # print('X cols - ',X.columns)
    # print('y cols - ',y.columns)
    
    print('data_encoded Crash Level - ',data_encoded['Crash_Level'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42)

    
    

    # Logistic Regression requires feature scaling, so let's scale our features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    with col2:
        vAR_st.info("Data Preprocessing Completed!")
        vAR_st.info("Classification Model Successfully Trained")

    # Create a Logistic Regression object
    log_reg = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)

    # Train the model
    log_reg.fit(X_train_scaled, y_train)
    
    print('LABELS - ',log_reg.classes_)
    
    return log_reg,X_train_raw,vAR_features



def Test_Model(vAR_test_data):
    
    vAR_col_index = []
    
    print('vAR_test_data cols before- ',vAR_test_data.columns)
    
    vAR_test_data = vAR_test_data[vAR_st.session_state['X_train_cols']]
    
    print('vAR_test_data cols after- ',vAR_test_data.columns)
    
    vAR_df_columns = vAR_test_data.columns
            
    vAR_numeric_columns = vAR_test_data._get_numeric_data().columns 
    
    vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
    
    vAR_model = vAR_st.session_state['vAR_model']
    
    # To fill categorical NaN column
    for col in vAR_categorical_column:
        vAR_col_index.append(vAR_df_columns.get_loc(col))
        vAR_test_data[col].fillna('Missing', inplace=True)
    
    # To fill Integer NaN column
    for col in vAR_numeric_columns:
        vAR_test_data[col].fillna(10000, inplace=True)
    
    # data_encoded = pd.get_dummies(vAR_train_df, columns=vAR_categorical_column)
    encoder = OrdinalEncoder()
    vAR_test_data[vAR_categorical_column] = encoder.fit_transform(vAR_test_data[vAR_categorical_column])
    
    data_encoded = vAR_test_data
    vAR_data_encoded_cols = data_encoded.columns

    print('type dataencoded - ',type(data_encoded))
    print('dataencoded cols - ',vAR_data_encoded_cols)
    
    data_encoded = data_encoded[vAR_st.session_state['X_train_cols']]
        
    print('vAR_numeric_columns test- ',vAR_numeric_columns)

    print('vAR_categorical_column test- ',vAR_categorical_column)
    
    print('data_encoded cols test- ',data_encoded.columns)
    
    # Logistic Regression requires feature scaling, so let's scale our features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(data_encoded)
    
    
    
                
    col1,col2,col3 = vAR_st.columns([3,15,1])
    
    with col2:
        
    
        # Predict probabilities on the test data
        y_pred_proba_log_reg = vAR_model.predict_proba(X_test_scaled)
        
        print('y_pred_proba_log_reg - ',y_pred_proba_log_reg)
        
        print('y_pred_proba_log_reg type- ',type(y_pred_proba_log_reg))

        # Convert to DataFrame for better visualization
        y_pred_proba_log_reg_df = pd.DataFrame(y_pred_proba_log_reg, columns=['INJURY CRASH','LOW-SEVERITY'])
        
        print('y_pred_proba_log_reg_df - ',y_pred_proba_log_reg_df)
        
        
        vAR_test_data["INJURY CRASH"] = y_pred_proba_log_reg_df["INJURY CRASH"]
        
        vAR_test_data["LOW-SEVERITY"] = y_pred_proba_log_reg_df["LOW-SEVERITY"]
        
        
        X_test_scaled_inversed = scaler.inverse_transform(X_test_scaled)
        
        print('X_test_scaled_inversed type- ',type(X_test_scaled_inversed))
        
        print('X_test_scaled_inversed shape- ',X_test_scaled_inversed.shape)
         
        X_test_new = encoder.inverse_transform(X_test_scaled_inversed[:,vAR_col_index]) 
        
        print('X_test_new - ',X_test_new)
        print('X_test_new type- ',type(X_test_new))
        
        # vAR_test_data[vAR_test_data[vAR_col_index]] = X_test_new
        
        for idx,cat_col in enumerate(vAR_categorical_column):
            
            vAR_test_data[cat_col] = X_test_new[:,idx]
        
        print('vAR_test_data newww- ',vAR_test_data)
        print('vAR_test_data newww type- ',type(vAR_test_data))
        
        if "vAR_test_data" not in vAR_st.session_state:
            vAR_test_data['LABEL'] = vAR_test_data[['INJURY CRASH', 'LOW-SEVERITY']].where(lambda x: x > 0.5).idxmax(axis=1)
            vAR_st.session_state["vAR_test_data"] = vAR_test_data
            # Ingest model outcome into Bigquery Table
            Upload_Data_To_GCS(vAR_test_data,"response")
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write(vAR_test_data)
        
        return vAR_test_data
                


def dataframe_to_base64(df):
    """Convert dataframe to base64-encoded csv string for downloading."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

def create_download_button(df, filename="data.csv"):
    """Generate a link to download the dataframe as a csv file."""
    b64_csv = dataframe_to_base64(df)
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="{filename}" style="display: block; margin: 1em 0; padding: 13px 20px 16px 12px; background-color: rgb(47 236 106); text-align: center; border: none; border-radius: 6px;color: black; text-decoration: none;">Download Model Outcome as CSV</a>'
    return href


def ModelOutcomeSummary(vAR_test_data):
    frequencies = vAR_test_data['LABEL'].value_counts()
    
    fig, ax = plt.subplots()
    bars = ax.bar(frequencies.index, frequencies.values, color=['blue', 'grey'])  # adjust colors as needed
    ax.set_xlabel('Label')
    ax.set_ylabel('Frequency')
    ax.set_title('Model Outcome Summary - Frequency of Crash Levels')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + 0.5,  # Adjust this value for the position of the count
                '%d' % int(height),
                ha='center', va='bottom')
    
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.pyplot(fig)

    
    
    
def ExplainableAI(X_train,features,vAR_model,vAR_test_data,test_id):
    print('train type - ',type(X_train))
    print('X_train - ',X_train.columns)
    print('vAR_test_data - ',type(vAR_test_data))
    print('dtypes - ',vAR_test_data.dtypes)
    print('test cols - ',vAR_test_data.columns)
    explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, 
                                              feature_names=features, 
                                              class_names=['INJURY CRASH', 'LOW-SEVERITY'], 
                                              mode='classification')
    
    exp = explainer_lime.explain_instance(
    vAR_test_data.iloc[int(test_id)], vAR_model.predict_proba,top_labels=1,num_features=8,num_samples=500)
    
    # Extract LIME explanations and visualize
    features, weights = zip(*exp.as_list())
    chart_data = pd.DataFrame({'Feature': features, 'Weight': weights})
    print(chart_data.head())
    
    
    
    vAR_st.dataframe(chart_data)
    
    vAR_st.write('')
    
    html = exp.as_html()
    components.html(html, height=1000,width=1000)




# def SHAPExplainableAI(X_train,vAR_model,X_test_data,test_id):
    
#     # SHAP
#     explainer = shap.LinearExplainer(vAR_model, X_train)
#     shap_values = explainer.shap_values(X_test_data)
    
#     # Force plot for selected instance
#     # shap.initjs()  # Required for visualizations
#     # plt.figure(figsize=(20, 3))
#     # shap.force_plot(explainer.expected_value, shap_values[test_id], X_test_data.iloc[test_id])
#     # vAR_st.pyplot(plt.gcf())  # Display the current figure

#     # Summary plot for global explanations
#     vAR_st.write("SHAP Summary Plot:")
#     shap.summary_plot(shap_values, X_test_data)  # Set `show=False` so it doesn't display immediately
#     vAR_st.pyplot(plt.gcf())  # Display the current figure




def SHAPExplainableAI(X_train,vAR_model,X_test_data,test_id):
    
    # Initialize the SHAP explainer for the trained model
    explainer = shap.Explainer(vAR_model, X_train)


    # Calculate SHAP values for the chosen instance
    shap_values = explainer(X_test_data)

    # Create a Streamlit app
    vAR_st.title("SHAP Explainer for Logistic Regression (Multi-class Classification)")

    # Display the SHAP summary plot
    vAR_st.set_option('deprecation.showPyplotGlobalUse', False)
    vAR_st.pyplot(shap.summary_plot(shap_values[0], X_test_data, plot_type="bar"))

    # Display the force plot for the chosen instance
    vAR_st.subheader("Force Plot for a Chosen Instance")
    vAR_st.set_option('deprecation.showPyplotGlobalUse', False)
    vAR_st.pyplot(shap.force_plot(explainer.expected_value, shap_values, X_test_data))


