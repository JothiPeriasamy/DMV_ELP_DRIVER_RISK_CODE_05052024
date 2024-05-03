import streamlit as vAR_st
from DSAI_GENAI.DSAI_Azure_Assistant import DataInsightsWithAssistant
from DSAI_Classification_Model.DSAI_Driver_Risk_Classification import Preview_Data
import pandas as pd
import os


AZURE_ASSISTANT_ID = os.environ["AZURE_ASSISTANT_ID"]

def GENAI_Analysis():



    vAR_dataset = None
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_result_data = pd.DataFrame()
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Upload Data')

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

        #Explore with GENAI
                
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        with col2:
            vAR_st.subheader('Interact with LLM(Prompt)')

            
        
        with col4:
            vAR_st.write("**Note:** LLM Response time may increase based on data volume.")
            vAR_user_input = vAR_st.text_area('Sample : Which court number has investigated more number of crashes?')
            vAR_submit = vAR_st.button("Submit")
        col1,col2,col3 = vAR_st.columns([1.5,10,1.5])

        if vAR_submit:   
            if vAR_user_input:
                vAR_llm_response = DataInsightsWithAssistant(vAR_user_input,AZURE_ASSISTANT_ID)

                with col2:
                    vAR_st.info(vAR_llm_response) 
        