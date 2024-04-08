import os
import json
from openai import AzureOpenAI
import streamlit as vAR_st
import time
    
client = AzureOpenAI(
    api_key=os.environ["API_KEY"],  
    api_version=os.environ["API_VERSION"],
    azure_endpoint = os.environ["AZURE_ENDPOINT"], 

    )




def DataInsightsWithAssistant(vAR_user_input,vAR_assistant_id):
    if "thread_id" not in vAR_st.session_state:
        vAR_st.session_state.thread_id = None

        
    thread = client.beta.threads.create()
    vAR_st.session_state.thread_id = thread.id
    
    vAR_response = run_assistant(vAR_user_input,vAR_st.session_state.thread_id,vAR_assistant_id)

    return vAR_response


def run_assistant(vAR_user_input,thread_id,vAR_assistant_id):
    
    # Add the user's message to the existing thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=vAR_user_input
    )
    
    # Create a run with additional instructions
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=vAR_assistant_id)
    
    
    while run.status != 'completed':
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
    # Retrieve messages added by the assistant
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    latest_message = messages.data[0]
    print('before response from llm - ',latest_message)
    # graph_image_file_id = latest_message.content[0].image_file.file_id
    response = latest_message.content[0].text.value
    
    # vAR_file_obj = client.files.retrieve_content(graph_image_file_id)
    
    # print('vAR_file_obj - ',vAR_file_obj)
    
    return response