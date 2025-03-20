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
from pymongo import MongoClient
import pymongo
import json
from bson import json_util, ObjectId
from datetime import datetime
import time
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from elasticsearch import Elasticsearch


def extract_technical_metadata(json_data):
    attributes = set()
    attribute_types = {}

    def process_item(item):
        if isinstance(item, dict):
            for attr, value in item.items():
                if isinstance(value, (dict, list)):
                    process_item(value)
                else:
                    attributes.add(attr)
                    attribute_types[attr] = type(value).__name__
        elif isinstance(item, list):
            for value in item:
                if isinstance(value, (dict, list)):
                    process_item(value)

    if isinstance(json_data, dict):
        process_item(json_data)
    elif isinstance(json_data, list):
        for item in json_data:
            process_item(item)

    num_items = count_items(json_data)

    technical_metadata = {
        'attributes': list(attributes),
        'attribute_types': attribute_types,
        'num_items': num_items
    }

    return technical_metadata

def extract_business_metadata(json_data, description, keywords):
    if isinstance(json_data, dict):
        description = json_data.get('description', '')
        keywords = json_data.get('keywords', [])
    else:
        description = description 
        keywords = keywords 

    business_metadata = {
        'description': description,
        'keywords': keywords
    }

    return business_metadata

def extract_operational_metadata(json_data, file_path, data_source):
    if isinstance(json_data, dict):
        source = data_source 
    else:
        source = data_source 

    last_modified = get_last_modified(file_path)

    operational_metadata = {
        'source': source,
        'last_modified': last_modified
    }

    return operational_metadata

def get_last_modified(file_path):
    if os.path.exists(file_path):
        timestamp = os.path.getmtime(file_path)
        last_modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    else:
        last_modified = ''

    return last_modified

def count_items(json_data):
    if isinstance(json_data, list):
        return len(json_data)
    else:
        return 1

def extract_administrative_metadata(organization_unit, business_unit, department, data_owner, domain_name):
    # Construct the business metadata dictionary from user inputs
    administrative_metadata = {
        "organization_unit": organization_unit,
        "business_unit": business_unit,
        "department": department,
        "data_owner": data_owner,
        "domain_name": domain_name
    }
    return administrative_metadata

def extract_semantics_metadata(json_data):
    concepts = set()
    relationships = []
    distribution = {}
    validity = {}
    quality = {}
    consistency = {}

    def process_item(item):
        if isinstance(item, dict):
            for attr, value in item.items():
                if isinstance(value, (dict, list)):
                    process_item(value)
                else:
                    concepts.add(attr)

    if isinstance(json_data, dict):
        process_item(json_data)
    elif isinstance(json_data, list):
        for item in json_data:
            process_item(item)

    concepts = list(concepts)
    for concept in concepts:
        values = [item[concept] for item in json_data if concept in item]

        if not values:
            # Skip calculations for attributes with no values
            continue

        if isinstance(values[0], str):
            # Skip calculations for string attributes
            continue

        # Calculate data distribution
        distribution[concept] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }

        # Calculate data validity and completeness
        validity[concept] = {
            'missing_values': sum(1 for value in values if value is None),
            'valid_values': sum(1 for value in values if value is not None),
            'completeness': sum(1 for value in values if value is not None) / len(json_data)
        }

        # Calculate data quality w.r.t. outliers
        if isinstance(values[0], (int, float)):
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            quality[concept] = {
                'outliers': sum(1 for z_score in z_scores if z_score > 3)
            }

        # Calculate data consistency w.r.t. distinct and unique values
        consistency[concept] = {
            'distinct_values': len(set(values)),
            'unique_values': len(np.unique(values))
        }

    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            source = concepts[i]
            target = concepts[j]
            relation = 'independent'

            values_source = [item[source] for item in json_data if source in item]
            values_target = [item[target] for item in json_data if target in item]

            try:
                if values_source and values_target:
                    if all(isinstance(value, (int, float)) for value in values_source) and all(isinstance(value, (int, float)) for value in values_target):
                        _, p_value = ttest_ind(values_source, values_target)
                        relation = 'independent' if p_value >= 0.05 else 'correlated'
                    elif all(not isinstance(value, (int, float)) for value in values_source) and all(not isinstance(value, (int, float)) for value in values_target):
                        observed = [[v1, v2] for v1, v2 in zip(values_source, values_target)]
                        try:
                            _, p_value, _, _ = chi2_contingency(observed)
                            relation = 'associated' if p_value >= 0.05 else 'not associated'
                        except (ValueError, TypeError):
                            relation = 'not associated'
            except (ValueError, TypeError):
                relation = 'not associated'

            relationships.append({
                'source': source,
                'target': target,
                'relation': relation
                  })

    semantics_metadata = {
        'concepts': concepts,
        'relationships': relationships,
        'distribution': distribution,
        'validity': validity,
        'quality': quality,
        'consistency': consistency
    }

    return semantics_metadata



class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JSONEncoder, self).default(obj)


def index_metadata(json_metadata, index_name):
    # Elasticsearch configurations

    es_host = "elasticsearch"
    es_port = 9200
    es_scheme = "http"  # Adjust the scheme as per your Elasticsearch setup
    es_index = str(index_name)

# Create Elasticsearch client
    es = Elasticsearch(
    [{"host": es_host, "port": es_port, "scheme": es_scheme}])

# Convert metadata to JSON format
    #json_metadata = json.dumps(json_metadata, indent=4)

    #es.index(index=index_name, body=json_metadata)
     # Check if the index exists
    if not es.indices.exists(index=index_name):
        # Create the index with the specified index name
        es.indices.create(index=index_name)

    # Index the metadata document
    es.index(index=index_name, body=json_metadata)

    st.success("Metadata indexed successfully.")




def app():
    image = Image.open('msis.jpeg')

    st.image(image, width=100)

    st.title('Welcome to Data Ingestion of NoSQL Data Types')

    st.sidebar.title('NoSQL DataTypes')
    option = st.sidebar.selectbox(
        'select subjects', ('MongoDB', 'Casaandra', 'DynamoDB', 'JSON/Avro/XML'))

    if option == 'MongoDB':
        minio_client = Minio('172.16.51.28:9000',
                             access_key='minio',
                             secret_key='miniostorage',
                             secure=False)
        client = MongoClient()
        mongo_host = st.text_input("Mongodb Host")
        # mongo_port = st.text_input("Mongdb Port")
        mongo_db = st.text_input("Mongdb Database")
        # mongo_collections = st.text_input("Mongdb Collections")
        connection_querry = "mongodb://" + mongo_host + ":27017/"
        myclient = pymongo.MongoClient(connection_querry)
        mydb = myclient[mongo_db]
        # mycol = mydb.mongo_collections
        cursor = mydb.list_collection_names()
        for collections in cursor:
            st.write(collections)
        options = []
        for (collection) in cursor:
            options.append(collection)
        collection_selected = st.multiselect("Collection List", options)
        for collection in collection_selected:
            mycol = mydb[collection]
            mycollcursor = mycol.find()
            for doc in mycollcursor:
                st.write(doc)
                data = json.loads(json_util.dumps(doc))
                filename = collection + ".json"
                start_time = datetime.now()
                with open(filename, "w+") as f:
                    json.dump(data, f)
                with open(filename, 'rb') as file_data:
                    file_stat = os.stat(filename)
                    minio_client.put_object(
                        'sree', filename, file_data, file_stat.st_size,)
                    st.write(f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

    if option == 'JSON/Avro/XML':
        st.sidebar.title('JSON Upload Types')
        option = st.sidebar.selectbox(
            'json_upload_types', ('File Browse', 'URL Link', 'Cloud Storage'))
        if option == 'File Browse':
            uploaded_file = st.file_uploader(
                "Choose a file", type=["json", "avro", "xml"])
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
                st.write(uploaded_file)
                # Read the uploaded JSON file
                file_contents = uploaded_file.read()
                try:
            # Parse the JSON data
                    json_file = json.loads(file_contents)

            # Perform operations with the JSON data
            # ...

            # Store the JSON data in a file
                    with open(uploaded_file.name, 'w') as file:
                        json.dump(json_file, file)

                    st.success("JSON file uploaded and stored successfully!")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please upload a valid JSON file.")
                # dataframe.to_csv(uploaded_file.name, index=False, header=True)
                #data = json.loads(json_util.dumps(uploaded_file))
                start_time = datetime.now()
                organization_unit = st.text_input("Organization Unit generating the data")
                business_unit = st.text_input("Business Unit of data")
                department = st.text_input("Departnemt name of data")
                domain_name = st.text_input("Domain of dataset")
                database_name = st.text_input("Database Name of dataset")
                data_owner= st.text_input("Owner of dataset")
                description = st.text_input("Description About Dataset")
                keywords = st.text_input("Keywords for catologing data")
                source = st.text_input("Source of dataset")
                index_name = st.text_input("Name of index of catologing data")
                #with open(uploaded_file.name, "w+") as f:
                #    json.dump(data, f)

                file_path = uploaded_file.name
                json_data = None

                # Read JSON file
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Try decoding as a dictionary
                try:
                    json_data = json.loads("".join(lines))
                    print("Dictionary structure detected.")
                except json.JSONDecodeError:
                    # Try decoding as an array
                    try:
                        json_data = [json.loads(line.strip()) for line in lines]
                        print("Array structure detected.")
                    except json.JSONDecodeError as e:
                        print("Error parsing JSON file:", str(e))
                        return

                # Extract technical metadata
                technical_metadata = extract_technical_metadata(json_data)

                # Extract business metadata
                business_metadata = extract_business_metadata(json_data, description, keywords)

                # Extract operational metadata
                operational_metadata = extract_operational_metadata(json_data, file_path, source)
                administrative_metadata = extract_administrative_metadata(organization_unit, business_unit, department,data_owner, domain_name)


                # Extract semantics metadata
                semantics_metadata = extract_semantics_metadata(json_data)
                # Create metadata dictionary
                metadata = {
                'technical_metadata': technical_metadata,
                'business_metadata': business_metadata,
                'operational_metadata': operational_metadata,
                'administrative_metadata': administrative_metadata,
                'semantics_metadata': semantics_metadata
                }

                # Dump the metadata into a new JSON file
                #with open(file_path, 'w') as file:

                 # Save the JSON metadata to a file
                #metadata_file_name = file_path + "_metadata.json" 
                metadata_file_name = file_path + "_metadata.json" 
                with open(metadata_file_name, 'w') as file:
                    json.dump(metadata, file, indent=4, cls=JSONEncoder)
                st.success(f"Metadata saved to {metadata_file_name}.")

                Catalog = "Catalogs"
                Data = "Data"
                client = Minio('172.16.51.28:9000',
                               access_key='minio',
                               secret_key='miniostorage',
                               secure=False)
                with open(metadata_file_name, 'rb') as file_data:
                    file_stat = os.stat(metadata_file_name)
                    client.put_object(
                        'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{database_name}/" + metadata_file_name, file_data, file_stat.st_size)

                with open(file_path, 'rb') as file_data:
                    file_stat = os.stat(file_path)
                    client.put_object(
                        'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{database_name}/" + file_path, file_data, file_stat.st_size)

                json_metadata = json.dumps(metadata, indent=4, cls=JSONEncoder)
                index_metadata(json_metadata, index_name)
                # st.write("Index of Document Created Succesfully")
                st.success("Metadata Indexed to Data Catloging Succesfully")


                st.write(f"my_app render: {(datetime.now() - start_time).total_seconds()}s")




        if option == 'URL Link':
            url = st.text_input("Provide URL Link of JSON Files")
            # url = 'http://google.com/favicon.ico'
            r = requests.get(url, allow_redirects=True)
            a = urlparse(url)
            st.write(a.path)
            st.write(os.path.basename(a.path))
            file_name_path = os.path.basename(a.path)
            data = json.loads(json_util.dumps(r))
            with open(file_name_path, "w+") as f:
                json.dump(data, f)
            client = Minio('172.16.51.28:9000',
                           access_key='minio',
                           secret_key='miniostorage',
                           secure=False)

            with open(file_name_path, 'rb') as file_data:
                file_stat = os.stat(file_name_path)

                start_time = datetime.now()
                client.put_object('sree', file_name_path, file_data,
                                  file_stat.st_size,)
                st.write(f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

        if option == 'Cloud Storage':
            st.sidebar.title('Cloud Storage Types')
            option = st.sidebar.selectbox(
                'select cloud_storage_types', ('Minio', 'S3 Bucket', 'Google Cloud Storage'))
            if option == 'Minio':
                a_key = st.text_input("Minio Access Key")
                s_key = st.text_input("Minio Secret Key")
                minio_host = st.text_input("Minio Host")
                connect_minio = minio_host + ":9000"
                client = Minio(connect_minio,
                               access_key=a_key,
                               secret_key=s_key,
                               secure=False)

                # List buckets
                buckets = client.list_buckets()
                for bucket in buckets:
                    # st.write('bucket:', bucket.name, bucket.creation_date)
                    st.write('bucket:', bucket.name)
                options = []
                for bucket in buckets:
                    options.append(bucket)
                bucket_selected = st.multiselect(
                    "Bucket List of Minio Storage", options)
                for mybucket in bucket_selected:
                    bucket_name = str(mybucket)
                    objects = client.list_objects(bucket_name,
                                                  recursive=True)
                    list_objects = []
                    for obj in objects:
                        # st.write(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified,
                        #         obj.etag, obj.size, obj.content_type)
                        st.write(obj.object_name, obj.size)
                        list_objects.append(obj.object_name)
                    objects_selected = st.multiselect(
                        "Objects Selected For Ingestion", list_objects)
                    for my_obj in objects_selected:
                        object_name = str(my_obj)
                        object_data = client.get_object(
                            bucket_name, object_name)
                        file_name = object_name
                        with open(file_name, 'wb') as file_data:
                            for d in object_data.stream(32*1024):
                                file_data.write(d)

                        with open(file_name, 'rb') as file_data:
                            file_stat = os.stat(file_name)
                            start_time = datetime.now()

                            client.put_object('msisbucket', file_name, file_data,
                                              file_stat.st_size,)
                            st.write(f"my_app render: {(datetime.now() - start_time).total_seconds()}s")
