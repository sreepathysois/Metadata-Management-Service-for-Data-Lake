import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import paramiko
import os
import pandas as pd
import csv
import mysql.connector
from io import StringIO
from minio import Minio
# from minio.error import ResponseError
import psycopg2
import requests
from urllib.parse import urlparse
import streamlit.components.v1 as stc
import shlex
# File Processing Pkgs
import pandas as pd
import docx2txt
from PIL import Image
from PyPDF2 import PdfFileReader
import pdfplumber
from PIL import Image
import subprocess
import json
from bson import json_util, ObjectId


def load_image(image_file):
    img = Image.open(image_file)
    return img


def app():
    image = Image.open('msis.jpeg')

    st.image(image, width=100)

    st.title(
        'Welcome to Data Ingestion of Medical Health Care Data Types From EHR EMR and Packs Server')

    st.sidebar.title('Healthcare DataTypes')
    option = st.sidebar.selectbox(
        'select subjects', ('EHR', 'Packs Diacom Data'))

    if option == 'EHR':

        st.sidebar.title('EHR Patient Data')
        option = st.sidebar.selectbox(
            'select text_upload_types', ('EHR', 'EMR'))
        if option == "EHR":
            st.subheader("HIS Data")
            client = Minio('172.16.51.28:9000',
                           access_key='minio',
                           secret_key='miniostorage',
                           secure=False)
            rest_server_url = st.text_input(
                "Enter URL to Register Client to Auth Server")
            app_name = st.text_input("Enter Client App Name")
            if st.button('Register'):
                rest_curl_post_cmd = '''  curl -X POST -k -H 'Content-Type: application/json' -i '''
                # rest_server_url = rest_server_url
                rest_register_data_cmd = ''' --data '{ "application_type": "private",
			"redirect_uris":
			["https://172.16.51.57:9300/swagger/oauth2-redirect.html"], '''
                rest_client_name = ''' "client_name":''' + ''' " ''' + app_name + ''' " , '''
                rest_register_scope = ''' "token_endpoint_auth_method": "client_secret_post",
			"contacts": ["me@example.org", "them@example.org"],
			"scope": "openid api:oemr api:fhir api:port user/allergy.read user/allergy.write user/appointment.read user/appointment.write user/dental_issue.read user/dental_issue.write user/document.read user/document.write user/drug.read user/encounter.read user/encounter.write user/facility.read user/facility.write user/immunization.read user/insurance.read user/insurance.write user/insurance_company.read user/insurance_company.write user/insurance_type.read user/list.read user/medical_problem.read user/medical_problem.write user/medication.read user/medication.write user/message.write user/patient.read user/patient.write user/practitioner.read user/practitioner.write user/prescription.read user/procedure.read user/soap_note.read user/soap_note.write user/surgery.read user/surgery.write user/vital.read user/vital.write user/AllergyIntolerance.read user/CareTeam.read user/Condition.read user/Encounter.read user/Immunization.read user/Location.read user/Medication.read user/MedicationRequest.read user/Observation.read user/Organization.read user/Organization.write user/Patient.read user/Patient.write user/Practitioner.read user/Practitioner.write user/PractitionerRole.read user/Procedure.read patient/encounter.read patient/patient.read patient/Encounter.read patient/Patient.read"
}' '''
                rest_register_cmd = rest_curl_post_cmd + rest_server_url + \
                    rest_register_data_cmd + rest_client_name + rest_register_scope
                st.write(rest_register_cmd)

                args = shlex.split(rest_register_cmd)
                process = subprocess.Popen(
                    args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                st.write(stdout)
                st.write(stderr)
            rest_authorize_url = st.text_input(
                'URL To authorize and get access token for Clients')
            client_id = st.text_input("Enter Client ID")
            admin = st.text_input("Enter Username")
            password = st.text_input("Enter Password")
            if st.button('Authorize Client and Get Access Token'):
                rest_curl_authorize_post = ''' curl -X POST -k -H 'Content-Type: application/x-www-form-urlencoded' -i '''
                rest_authorize_url = rest_authorize_url
                rest_request_access_token_data = ''' --data 'grant_type=password'''
                rest_client_id = '''&client_id=''' + client_id + ''' ''' + '''&user_role=users'''
                rest_authorize_credentials = '''&username=''' + \
                    admin + '''&password=''' + password
                rest_authorize_scope = '''&scope=openid api:oemr api:fhir api:port user/allergy.read user/allergy.write user/appointment.read user/appointment.write user/dental_issue.read user/dental_issue.write user/document.read user/document.write user/drug.read user/encounter.read user/encounter.write user/facility.read user/facility.write user/immunization.read user/insurance.read user/insurance.write user/insurance_company.read user/insurance_company.write user/insurance_type.read user/list.read user/medical_problem.read user/medical_problem.write user/medication.read user/medication.write user/message.write user/patient.read user/patient.write user/practitioner.read user/practitioner.write user/prescription.read user/procedure.read user/soap_note.read user/soap_note.write user/surgery.read user/surgery.write user/vital.read user/vital.write user/AllergyIntolerance.read user/CareTeam.read user/Condition.read user/Encounter.read user/Immunization.read user/Location.read user/Medication.read user/MedicationRequest.read user/Observation.read user/Organization.read user/Organization.write user/Patient.read user/Patient.write user/Practitioner.read user/Practitioner.write user/PractitionerRole.read user/Procedure.read patient/encounter.read patient/patient.read patient/Encounter.read patient/Patient.read ' '''

                rest_authorize_token_cmd = rest_curl_authorize_post + rest_authorize_url + \
                    rest_request_access_token_data + rest_client_id + \
                    rest_authorize_credentials + rest_authorize_scope
                st.write(rest_authorize_token_cmd)
                args = shlex.split(rest_authorize_token_cmd)
                process = subprocess.Popen(
                    args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                st.write(stdout)
                st.write(stderr)
            options = ["api/patient", "fhir/Patient"]
            api_selected = st.multiselect("API List", options)
            for api in api_selected:
                #rest_api = st.text_input("Enter REST API")
                rest_api = api
                rest_api_patient = ''' curl --insecure -X GET 'https://172.16.51.57:9300/apis/default/'''
                bearer = '''' -H 'Authorization: Bearer '''
                auth_token = st.text_input(
                    "Enter Access Token to Call REST API")
                if st.button("REST API to Fetch Data from ERP Server"):
                    api_cmd = rest_api_patient + rest_api + bearer + auth_token + ''' ' '''
                    st.write(api_cmd)
                    args = shlex.split(api_cmd)
                    process = subprocess.Popen(
                        args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    st.write(stdout)
                    filename = rest_api.split("/")[1] + ".json"
                # st.write(stderr)
                    #tempDir = "/home/msis/sreephd_data_ingestion_service/."
                    tempDir = os.getcwd() 
             # filename = api.split("/")[1] + ".json"
                    with open(os.path.join(tempDir, filename), "wb") as f:
                        f.write(stdout)

                # data = json.loads(json_util.dumps(stdout))
                # filename = api.split("/")[1] + ".json"
                # with open(filename, "w+") as f:
                #	json.dump(data, f)
                    with open(filename, 'rb') as file_data:
                        file_stat = os.stat(filename)
                    # st.write(file_stat)
                        client = Minio('172.16.51.28:9000',
                                       access_key='minio',
                                       secret_key='miniostorage',
                                       secure=False)
                        client.put_object(
                            'sree', filename, file_data, file_stat.st_size)

    if option == 'Packs Diacom Data':
        st.sidebar.title('Diacom Image Upload Types')
        option = st.sidebar.selectbox(
            'select image_upload_types', ('Pacs Server Rest API', 'URL Link', 'Object Storage Types'))
        if option == 'Pacs Server Rest API':
            st.subheader("Diacom Image Dataset")
            client = Minio('172.16.51.28:9000', access_key='minio',
                           secret_key='miniostorage', secure=False)
            st.subheader("Image")
            pacs_server_url = st.text_input("Enter PACS Server REST API URL")
            admin = st.text_input("Enter PACS Server Admin Name")
            password = st.text_input("Enter PACS Server Password")
            user_credentials = admin + ":" + password
            get_request_url = ''' --request GET --url '''
            get_patient_ids = ''' curl --user ''' + user_credentials + \
                get_request_url + pacs_server_url + '''/patients'''
            patient_list_cmd = get_patient_ids

            args = shlex.split(patient_list_cmd)
            process = subprocess.Popen(
                args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            # print(stdout)
            patient = stdout.decode('UTF-8')
            # print(patient)
            patient_id = patient.split('"')[1]
            st.write(patient_id)
            patient_list = []
            patient_list.append(patient_id)
            st.write(patient_list)
            patient_selected = st.multiselect("Patient List Ids", patient_list)
            for patient_id in patient_selected:
                get_study_ids = ''' curl --user ''' + user_credentials + \
                    get_request_url + pacs_server_url + '''/patients/''' + patient_id
                st.write(get_study_ids)
                args = shlex.split(get_study_ids)
                process = subprocess.Popen(
                    args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                st.write(stdout)
                study_json = stdout.decode('utf8')
                study_data = json.loads(study_json)
                # print(study_data)
                # print(type(study_data))
                # print(study_data["Studies"])
                study_id_list = []
                study_id = study_data["Studies"]
                st.write(study_id[0])
                # study_id_list.append(study_id)
                st.write(study_id)
                study_selected = st.multiselect(
                    "Study Selected for Data Ingestion", study_id)
                for study in study_selected:
                    get_series_ids = ''' curl --user ''' + user_credentials + \
                        get_request_url + pacs_server_url + '''/studies/''' + study
                    st.write(get_series_ids)
                    args = shlex.split(get_series_ids)
                    process = subprocess.Popen(
                        args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    st.write(stdout)
                    series_json = stdout.decode('utf8')
                    series_data = json.loads(series_json)
                    # print(study_data)
                    # print(type(study_data))
                    # print(study_data["Studies"])
                    series_id = series_data["Series"]
                    st.write(series_id)
                    series_selected = st.multiselect(
                        "Series Selected For Data Ingestion", series_id)
                    for series in series_selected:
                        get_instance_ids = ''' curl --user ''' + user_credentials + \
                            get_request_url + pacs_server_url + '''/series/''' + series
                        st.write(get_instance_ids)
                        args = shlex.split(get_instance_ids)
                        process = subprocess.Popen(
                            args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = process.communicate()
                        st.write(stdout)
                        instance_json = stdout.decode('utf8')
                        instance_data = json.loads(instance_json)
                        # print(study_data)
                        # print(type(study_data))
                        # print(study_data["Studies"])
                        instance_id = instance_data["Instances"]
                        st.write(instance_id)
                        instance_selected = st.multiselect(
                            "Instance Selected to Ingest Dicom Images", instance_id)
                        for instance in instance_selected:
                            filename = instance + ".png"
                            get_instance_data = ''' curl --user ''' + user_credentials + get_request_url + \
                                pacs_server_url + '''/instances/''' + \
                                instance + '''/preview -o  ''' + filename
                            args = shlex.split(get_instance_data)
                            process = subprocess.Popen(
                                args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            stdout, stderr = process.communicate()
                            st.write(stdout)
                            with open(filename, 'rb') as file_data:
                                file_stat = os.stat(str(filename))
                                client.put_object(
                                    'sree', f"{patient_id}/{study}/{series}/" + filename, file_data , file_stat.st_size)

    if option == 'Videos':
        st.sidebar.title('Video Upload Types')
        option = st.sidebar.selectbox(
            'select video_upload_types', ('Local File Browse', 'URL Link', 'Object Storage Types'))
        if option == 'Local File Browse':
            st.subheader("Video Dataset")
            client = Minio('172.16.51.28:9000', access_key='minio',
                           secret_key='miniostorage', secure=False)
            st.subheader("Image")
            uploaded_files = st.file_uploader(
                "Upload Images", type=["mp4", "mpeg"], accept_multiple_files=True)

            if uploaded_files is not None:
                for video_file in uploaded_files:
                    file_details = {"filename": video_file.name, "filetype": video_file.type,
                                    "filesize": video_file.size}
                    st.write(file_details)
                    st.video(video_file)
                    fileDir = "/home/msis/sreephd_data_ingestion_service/."
                    with open(os.path.join(fileDir, video_file.name), "wb") as f:
                        f.write((video_file).getbuffer())

                    with open(file_details['filename'], 'rb') as file_data:
                        file_stat = os.stat(file_details['filename'])
                        client.put_object(
                            'sree', file_details['filename'], file_data, file_stat.st_size)
