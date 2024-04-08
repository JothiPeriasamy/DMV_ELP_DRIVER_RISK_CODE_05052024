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


import os
from google.cloud import bigquery,storage
import datetime

def RequestToBigquery(vAR_request_df):
    vAR_client = bigquery.Client()
    vAR_request_table_name = "DMV_DRP_DRIVER_RISK_REQUEST"
    
    vAR_len = len(vAR_request_df)

    created_at = []
    created_by = []
    updated_at = []
    updated_by = []
    created_at += vAR_len * [datetime.datetime.utcnow()]
    created_by += vAR_len * [os.environ['GCP_USER']]
    updated_by += vAR_len * [os.environ['GCP_USER']]
    updated_at += vAR_len * [datetime.datetime.utcnow()]

    vAR_request_df['CREATED_USER'] = created_by
    vAR_request_df['CREATED_DT'] = created_at
    vAR_request_df['UPDATED_USER'] = updated_by
    vAR_request_df['UPDATED_DT'] = updated_at



    # Define table name, in format dataset.table_name
    table = os.environ["GCP_BQ_SCHEMA_NAME"]+'.'+vAR_request_table_name
    job_config = bigquery.LoadJobConfig(autodetect=True,write_disposition="WRITE_APPEND",source_format=bigquery.SourceFormat.CSV)
    # job_config = bigquery.LoadJobConfig(autodetect=True,write_disposition="WRITE_APPEND",source_format=bigquery.SourceFormat.CSV,max_bad_records=vAR_number_of_configuration,allowJaggedRows=True)
    job = vAR_client.load_table_from_dataframe(vAR_request_df, table,job_config=job_config)

    job.result()  # Wait for the job to complete.
    table_id = os.environ['GCP_PROJECT_ID']+'.'+table
    table = vAR_client.get_table(table_id)  # Make an API request.
    print(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), table_id
            )
        )



def TestDataResponseToBigquery(vAR_model_result_df):
    vAR_client = bigquery.Client()
    vAR_response_table_name = "DMV_DRP_DRIVER_RISK_MODEL_RESPONSE"
    
    vAR_len = len(vAR_model_result_df)

    created_at = []
    created_by = []
    updated_at = []
    updated_by = []
    created_at += vAR_len * [datetime.datetime.utcnow()]
    created_by += vAR_len * [os.environ['GCP_USER']]
    updated_by += vAR_len * [os.environ['GCP_USER']]
    updated_at += vAR_len * [datetime.datetime.utcnow()]

    vAR_model_result_df['CREATED_USER'] = created_by
    vAR_model_result_df['CREATED_DT'] = created_at
    vAR_model_result_df['UPDATED_USER'] = updated_by
    vAR_model_result_df['UPDATED_DT'] = updated_at



    # Define table name, in format dataset.table_name
    table = os.environ["GCP_BQ_SCHEMA_NAME"]+'.'+vAR_response_table_name
    job_config = bigquery.LoadJobConfig(autodetect=True,write_disposition="WRITE_APPEND",source_format=bigquery.SourceFormat.CSV)
    # job_config = bigquery.LoadJobConfig(autodetect=True,write_disposition="WRITE_APPEND",source_format=bigquery.SourceFormat.CSV,max_bad_records=vAR_number_of_configuration,allowJaggedRows=True)
    job = vAR_client.load_table_from_dataframe(vAR_model_result_df, table,job_config=job_config)

    job.result()  # Wait for the job to complete.
    table_id = os.environ['GCP_PROJECT_ID']+'.'+table
    table = vAR_client.get_table(table_id)  # Make an API request.
    print(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), table_id
            )
        )





def InsertErrorLog(vAR_err_code,vAR_err_context):
    
    vAR_err_code = vAR_err_code.replace('"',"'")
    vAR_err_context = vAR_err_context.replace('"',"'")
    
    vAR_suppress_key_err1 = "st.session_state has no key 'training_data'"
    vAR_suppress_key_err2 = "st.session_state has no key 'vAR_model'"
    vAR_suppress_key_err3 = "st.session_state has no key 'vAR_tested_log'"
    
    if (vAR_suppress_key_err1 in vAR_err_code) or (vAR_suppress_key_err2 in vAR_err_code) or (vAR_suppress_key_err3 in vAR_err_code): 
        print("Key Error Suppressed...")
        pass
    else:
         
        # Load client
        client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
        table = os.environ["GCP_BQ_SCHEMA_NAME"]+'.DMV_DRP_DRIVER_RISK_ERROR_LOG'
        
        vAR_query = """
        insert into `{}`(ERROR_CONTEXT,ERROR_CODE) values("{}","{}")
        """.format(table,vAR_err_code,vAR_err_context)
        
        print('Insert response table query - ',vAR_query)

        vAR_job = client.query(vAR_query)
        vAR_job.result()
        
        print("Error Inserted into Error Log Table")
        
        
        
        
        
def Upload_Data_To_GCS(vAR_data,type):
    vAR_request = vAR_data.to_csv()
    vAR_bucket_name = os.environ['GCS_BUCKET_NAME']
    vAR_utc_time = datetime.datetime.utcnow()
    client = storage.Client()
    bucket = client.get_bucket(vAR_bucket_name)
    if type.lower()=="request":
        vAR_file_path = os.environ["GCP_REQUEST_PATH"]+'/'+vAR_utc_time.strftime('%Y%m%d')+'/'+vAR_utc_time.strftime('%H%M%S')+".csv"
    else:
        vAR_file_path = os.environ["GCP_RESPONSE_PATH"]+'/'+vAR_utc_time.strftime('%Y%m%d')+'/'+vAR_utc_time.strftime('%H%M%S')+".csv"
    vAR_file_path = vAR_file_path.lower()
    bucket.blob(vAR_file_path).upload_from_string(vAR_request, 'text/csv')
    print('DMV DRP Request successfully saved into cloud storage')
    print('Path - ',vAR_file_path)
    return vAR_file_path


