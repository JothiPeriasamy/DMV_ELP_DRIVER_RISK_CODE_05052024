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




import streamlit as st
from PIL import Image


def CSS_Property(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def All_Initialization():
    image = Image.open('DSAI_Utility/Logo_final.png')
    st.image(image)
    st.markdown("<h1 style='text-align: center; color: #454545; font-size:25px;'>DMV Driver Risk Level Classification Application</h1><h2 style='text-align: center; color: blue; font-size:20px;'>Using LLM and a Classification Model</h2>", unsafe_allow_html=True)
    st.markdown("""
    <hr style="width:100%;height:3px;background-color:gray;border-width:10">
    """, unsafe_allow_html=True)
    choice1 =  st.sidebar.selectbox(" ",('Home','About Us'))
    choice2 =  st.sidebar.selectbox(" ",('Libraries in Scope','Vertex AI','Pandas','Streamlit','OS','Json'))
    choice3 =  st.sidebar.selectbox(" ",('Models Used',"Linear Regression",'Logistic Regression'))
    menu = ["Google Cloud Services in Scope","Cloud Storage", "Cloud Run", "Cloud Function", "Secret Manager"]
    choice = st.sidebar.selectbox(" ",menu)
    st.sidebar.write('')
    st.sidebar.write('')
    href = """<form action="#">
    <input type="submit" value="Clear/Reset" />
</form>"""
    st.sidebar.markdown(href, unsafe_allow_html=True)

    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.text('Build & Deployed on')
    st.sidebar.write('')
    col1,col2,col3 = st.sidebar.columns([8,0.1,8])
    with col1:
        st.image('DSAI_Utility/Google-Cloud-Platform-GCP-logo.png')
    with col3:
        # st.image('DSAI_Utility/openai-logo.png')
        st.image('DSAI_Utility/chatgpt-icon.png')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    
    # vAR_clear_button = st.sidebar.button('Clear/Reset')
    # if vAR_clear_button:
    #     st.experimental_rerun()
