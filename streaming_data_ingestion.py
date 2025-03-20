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
import boto3
import subprocess
import shlex


def app():
    image = Image.open('msis.jpeg')

    st.image(image, width=100)

    st.title('Welcome to Data Ingestion of Streaming Data ')

    st.sidebar.title('DataTypes')
    option = st.sidebar.selectbox(
        'select subjects', ('ERP/CRM Streams', 'IoT Stream Data', 'Social Media', 'Healthcare Stream Data'))

    if option == 'ERP/CRM Streams':

        st.sidebar.title('Databases Types')
        option = st.sidebar.selectbox(
            'select databases_types', ('Kafka', 'Firebase', 'Spark Streams'))
        if option == 'Kafka':
            kafka_topic_name = st.text_input(
                "Enter Kafka Topic for Streaming Data")
            s3_bucket_name = st.text_input(
                "Enter Bucket Name to Ingest Stream Data")
            #aws_access_key_id = st.text_input("Enter Object Storage Access Key Id")
            #aws_secret_access_key_id = st.text_input("Enter Object Storage Secret Access Key Id")
            aws_access_key_id = "AKIAY3QJIBZEI6VI4GUB"
            aws_secret_access_key_id = "atg08pOphm4K0aZJCwWfNuSHamHdfJlVnkyqF1r6"
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
            ssh.connect("172.16.51.135", username="kafka",
                        password="kafka@123", port=22)
            #topic = "sree_my_topic"
            kafka_topic_cmd = "sed -i '/topics=*/c\ topics=" + \
                kafka_topic_name + "'" + " /home/kafka/plugins/s3-sink.properties"
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
            ssh.connect("172.16.51.135", username="kafka",
                        password="kafka@123", port=22)
            # bucket_name_cmd = "sed -i '6s/s3\.bucket\.name=*/c\ s3.bucket.name=" + \
            bucket_name_cmd = "sed -i -r 's/s3.bucket.name=*/s3.bucket.name=" + \
                s3_bucket_name + "/'" + " " + "/home/kafka/plugins/s3-sink.properties"
            s3_bucket_name + "'" + " /home/kafka/plugins/s3-sink.properties"
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                bucket_name_cmd)

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
            ssh.connect("172.16.51.135", username="kafka",
                        password="kafka@123", port=22)
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                kafka_topic_cmd)
            st.write("Created Kafka Topic Succesfully")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
            ssh.connect("172.16.51.135", username="kafka",
                        password="kafka@123", port=22)
            # connector_name_cmd = "sed -i '1s/name=/c\ name=" + \
            #    kafka_topic_name + "'" + " /home/kafka/plugins/s3-sink.properties"
            connector_name_cmd = "sed -i -r 's/name=.*/name=" + kafka_topic_name + \
                "/'" + " " + "/home/kafka/plugins/s3-sink.properties"
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                connector_name_cmd)
            exit_code = ssh_stdout.channel.recv_exit_status()  # handles async exit error
            # st.write(ssh_stdout)
            #st.write("Created Connector Properties Configuration")
            kafka_topic_create_cmd = "~/kafka/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic " + kafka_topic_name
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                kafka_topic_create_cmd)
            #st.write("Create Kafka Topic on Host and Zookeeper and Broker Started")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
            ssh.connect("172.16.51.135", username="kafka",
                        password="kafka@123", port=22)
            aws_access_credentials_set_cmd = "sed -i '/aws_access_key_id=*/c\ aws_access_key_id= " + \
                aws_access_key_id + "'" + " /home/kafka/.aws/credentials"
            aws_secret_credentials_set_cmd = "sed -i '/aws_secret_access_key=*/c\ aws_secret_access_key= " + \
                aws_secret_access_key_id + "'" + " /home/kafka/.aws/credentials"
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                aws_access_credentials_set_cmd)
            #st.write("User Credentials Loaded Sucessfully")
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                aws_secret_credentials_set_cmd)
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key_id,
                region_name='us-east-1')
            # Then use the session to get the resource
            s3 = session.resource('s3')
            s3.create_bucket(Bucket=s3_bucket_name)
            #st.write("Bucket Created Sucessfully")
            if st.button('Start Connector and Stream Data'):
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
                ssh.connect("172.16.51.135", username="kafka",
                            password="kafka@123", port=22)
                kafka_s3_connector_cmd = "~/kafka/bin/connect-standalone.sh   ~/plugins/connector.properties   ~/plugins/s3-sink.properties"
                st.write(kafka_s3_connector_cmd)
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                    kafka_s3_connector_cmd)
                st.write(ssh_stdout)
                st.write(ssh_stderr)
                connector_status_cmd = ''' curl http://172.16.51.135:8083/connectors/''' + \
                    kafka_topic_name + '''/status'''
                args = shlex.split(connector_status_cmd)
                process = subprocess.Popen(
                    args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()

                st.write("Kafka Topic Created Sucessfully")
                st.write("S3 Bucket Created Sucessfully")
                st.write("Kafka Zookeeper and Broker Created Sucessfully")
                st.write("Kafka Connector Starting")
                st.write(stdout)

    if option == 'CSV':
        st.sidebar.title('CSV Upload Types')
        option = st.sidebar.selectbox(
            'csv_upload_types', ('File Browse', 'URL Link', 'Cloud Storage'))
        if option == 'File Browse':
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                # bytes_data = uploaded_file.getvalue()
                # st.write(bytes_data)

                # To convert to a string based IO:
                # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # st.write(stringio)

                # To read file as string:
                # string_data = stringio.read()
                # st.write(string_data)

                # Can be used wherever a "file-like" object is accepted:
                st.write("Filename: ", uploaded_file.name)
                dataframe = pd.read_csv(uploaded_file)
                st.write(dataframe)
                dataframe.to_csv(uploaded_file.name, index=False, header=True)
                client = Minio('172.16.51.28:9000',
                               access_key='minio',
                               secret_key='miniostorage',
                               secure=False)

                with open(uploaded_file.name, 'rb') as file_data:
                    file_stat = os.stat(uploaded_file.name)

                    client.put_object('sree', uploaded_file.name, file_data,
                                      file_stat.st_size,)
        if option == 'URL Link':
            url = st.text_input("Provide URL Link of CSV Files")
            # url = 'http://google.com/favicon.ico'
            r = requests.get(url, allow_redirects=True)
            a = urlparse(url)
            st.write(a.path)
            st.write(os.path.basename(a.path))
            file_name_path = os.path.basename(a.path)
            client = Minio('172.16.51.28:9000',
                           access_key='minio',
                           secret_key='miniostorage',
                           secure=False)
            # EDIT - because comment.
            df = pd.read_csv(url)
            df.to_csv(file_name_path, index=False, header=True)
            with open(file_name_path, 'rb') as file_data:
                file_stat = os.stat(file_name_path)

                client.put_object('sree', file_name_path, file_data,
                                  file_stat.st_size,)

        if option == 'Cloud Storage':
            st.sidebar.title('Cloud Storage Types')
            option = st.sidebar.selectbox(
                'select cloud_storage_types', ('Minio', 'S3 Bucket', 'Google Cloud Storage'))
            if option == 'S3 Bucket':
                minio_client = Minio('172.16.51.28:9000',
                                     access_key='minio',
                                     secret_key='miniostorage',
                                     secure=False)
                # aws_access_key = st.text_input("AWS Access Key")
                # aws_secret_key = st.text_input("AWS Secret Key")
                # aws_region_name = st.text_input("AWS Region Code")
                session = boto3.Session(
                    aws_access_key_id='',
                    aws_secret_access_key='',
                    region_name='us-east-1')
                # Then use the session to get the resource
                s3 = session.resource('s3')
                buckets_list = s3.buckets.all()
                for bucket in buckets_list:
                    st.write(bucket.name)
                list_buckets = []
                for bucket in buckets_list:
                    list_buckets.append(bucket.name)
                bucket_selected = st.multiselect("Buckets List", list_buckets)
                for mybucket in bucket_selected:
                    st.write("Bucket Selected For Ingestion of Objects", mybucket)
                for mybucket in bucket_selected:
                    my_bucket = s3.Bucket(mybucket)
                    # for my_bucket_object in my_bucket.objects.all():
                    #    st.write(my_bucket_object.key)
                    #    filename = my_bucket_object.key
                    #    my_bucket.download_file(my_bucket_object.key, filename)

                    list_bucket_objects = []
                    my_bucket = s3.Bucket(mybucket)
                    for bucket_object in my_bucket.objects.all():
                        list_bucket_objects.append(bucket_object.key)
                    object_selected = st.multiselect(
                        "Objects of Bucket Selected for Ingestion", list_bucket_objects)
                    for bucket_object in object_selected:
                        # s3.Bucket(my_bucket).download_file(object.key, bucket_object.key)
                        my_bucket.download_file(
                            my_bucket_object.key, my_bucket_object.key)
                        with open(my_bucket_object.key, 'rb') as file_data:
                            file_stat = os.stat(my_bucket_object.key)
