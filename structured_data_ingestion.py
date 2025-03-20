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
from datetime import datetime
import json
from elasticsearch import Elasticsearch
import csv
import spacy
import numpy as np
from scipy.stats import pearsonr, chi2_contingency, f_oneway, ttest_ind
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal
import re
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt
import io


def extract_technical_metadata(csv_file, secure_fields):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract the field names
    field_names = df.columns.tolist()

    # Extract the data types
    data_types = df.dtypes.tolist()

    # Create a list of dictionaries for field information
    fields = []
    for field_name, data_type in zip(field_names, data_types):
        field = {"name": field_name, "data_type": str(data_type)}
        if field_name in secure_fields:
            field["security"] = "Private"
        else:
            field["security"] = "Public"

        fields.append(field)

    # Count the number of records
    num_records = len(df)

    # Create the technical metadata dictionary
    technical_metadata = {
        "fields": fields,
        "num_records": num_records
    }

    return technical_metadata


def extract_business_metadata(description, database_name, keywords):
    # Construct the business metadata dictionary from user inputs
    business_metadata = {
        "description": description,
        "database_name": database_name,
        "keywords": keywords.split(",")
    }

    return business_metadata


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


def extract_operational_metadata(source):
    # Construct the operational metadata dictionary from user input
    operational_metadata = {
        "source": source
    }

    return operational_metadata


def extract_semantics_metadata(file_name):
    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Clean the data by dropping rows with missing values and replacing infinite values with NaN
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Extract the column names as concepts
    concepts = list(df_cleaned.columns)

    # Analyze relationships between fields
    relationships = []

    # Perform data distribution analysis
    data_distribution = df.describe().to_dict()

    # Perform data validity checks for completeness
    data_validity = {}
    for concept in concepts:
        completeness = df[concept].notnull().sum() / len(df)
        data_validity[concept] = {
            'completeness': completeness
        }

    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            concept1 = concepts[i]
            concept2 = concepts[j]

            # Perform correlation analysis for numeric variables
            if df_cleaned[concept1].dtype == 'float64' and df_cleaned[concept2].dtype == 'float64':
                correlation, _ = pearsonr(
                    df_cleaned[concept1], df_cleaned[concept2])

                if not np.isnan(correlation):
                    relationship = {
                        'source': concept1,
                        'target': concept2,
                        'relation': 'correlated',
                        'correlation': correlation
                    }
                    relationships.append(relationship)

            # Perform association analysis for categorical variables
            elif df_cleaned[concept1].dtype == 'object' and df_cleaned[concept2].dtype == 'object':
                contingency_table = pd.crosstab(
                    df_cleaned[concept1], df_cleaned[concept2])
                chi2, p, _, _ = chi2_contingency(contingency_table)

                if p < 0.05:
                    relationship = {
                        'source': concept1,
                        'target': concept2,
                        'relation': 'associated'
                    }
                    relationships.append(relationship)

            # Perform independence analysis for other variable types
            else:
                independence_test, p = chi2_independence_test(
                    df_cleaned[concept1], df_cleaned[concept2])

                if p < 0.05:
                    relationship = {
                        'source': concept1,
                        'target': concept2,
                        'relation': 'independent'
                    }
                    relationships.append(relationship)

     # Perform data quality checks for outliers
    data_quality = {}
    # Missing Values
    missing_values = df.isnull().sum().to_dict()
    data_quality['missing_values'] = missing_values

    for concept in concepts:
        if df[concept].dtype == 'float64':
            q1 = df[concept].quantile(0.25)
            q3 = df[concept].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[concept] < lower_bound) |
                          (df[concept] > upper_bound)]
            data_quality[concept] = {
                'outliers': len(outliers)
            }

    # Perform data consistency checks
    # data_consistency = {}
    """
    for concept in concepts:
        unique_values = df[concept].unique()
        if len(unique_values) > 1:
            data_consistency[concept] = {
                'distinct_values': unique_values.tolist()
            }

    consistency_validation = {}

    for concept in concepts:
        column_data = df[concept]

        # Generic data validation consistency check
        unique_values = column_data.unique()
        if len(unique_values) == 1:
            consistency_validation[concept] = 'Constant'
        elif len(unique_values) == len(column_data):
            consistency_validation[concept] = 'Unique'
        else:
            consistency_validation[concept] = 'Mixed'
    # Create the ontology metadata dictionary
    """
    data_consistency_validation = {}

    for column in df.columns:
        unique_count = df[column].nunique()
        distinct_values_count = len(df[column].unique())
        distinct_values = df[column].unique().tolist()

        data_consistency_validation[column] = {
            'unique_count': unique_count,
            'distinct_count': distinct_values_count
            # 'distinct_values': distinct_values
        }

    semantics_metadata = {
        'concepts': concepts,
        'relationships': relationships,
        'data_distribution': data_distribution,
        'data_validity': data_validity,
        'data_quality': data_quality,
        'data_consistency': data_consistency_validation
    }

    return semantics_metadata


def chi2_independence_test(variable1, variable2):
    contingency_table = pd.crosstab(variable1, variable2)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p


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

    st.title('Welcome to Data Ingestion of Structured Data Types')

    st.sidebar.title('DataTypes')
    option = st.sidebar.selectbox(
        'select subjects', ('Databases', 'Data Warehouses', 'CSV', 'Excel'))

    if option == 'Databases':

        st.sidebar.title('Databases Types')
        option = st.sidebar.selectbox(
            'select databases_types', ('Mysql', 'Postgres SQL'))
        if option == 'Mysql':
            host_name = st.text_input("Database Server Name")
            db_user_name = st.text_input("Database User Name")
            db_user_password = st.text_input("Database User Password")
            db_name = st.text_input("Database Name")
            st.write(db_user_name)
            mydb = mysql.connector.connect(
                host=host_name,
                user=db_user_name,
                password=db_user_password,
                # database=sys.argv[3]
                database=db_name, auth_plugin='mysql_native_password')  # Name of t
            cursor = mydb.cursor()
            tables_display_querry = "show tables from" + " " + db_name
            st.write(tables_display_querry)
            sql = tables_display_querry
            # sql = "INSERT INTO virtuallabs.student VALUES (%s,%s,%s,%s,%s,%s)"
            cursor.execute(tables_display_querry)
            tables = cursor.fetchall()
            # for table_name in cursor:
            # st.write(table_name)
            options = []
            for (table_name,) in tables:
                options.append(table_name)
            tables_selected = st.multiselect("Tables List", options)
            # mydb.commit()
            # st.table(table_list)
            st.write(tables_selected)
            # st.write(tables_selected[1])
            for tables in tables_selected:
                st.write(tables)
            st.write(
                "Export Tables of Your Choice in to CSV for Ingestion to Data Lake ")
            for table in tables_selected:
                table_export_querry = "select * from " + " " + db_name + "." + table
                sql_query = pd.read_sql_query(table_export_querry, mydb)
                df = pd.DataFrame(sql_query)
                st.table(df.head())
                # file_path_name = "/home/msis/sreephd_data_ingestion_service/dis_version_1_ingestion_hetrogenous_types/" + table + "." + "csv"
                start_time = datetime.now()
                file_path_name = os.getcwd() + "/" + table + "." + "csv"
                df.to_csv(file_path_name, index=True)
                object_name = table + "." + "csv"
                organization_unit = st.text_input("Organization Unit Name")
                business_unit = st.text_input("Business Unit Name")
                department = st.text_input("Department Name")
                data_owner = st.text_input("Data Owner Name")
                domain_name = st.text_input("Data Domain/Context Name")
                database_name = st.text_input("Database Name of Dataset")
                description = st.text_input("Description About DataSet")
                source = st.text_input("Source of DataSet")
                keywords = st.text_input("Keywords of Dataset")
                index_name = st.text_input(
                    "Index Name of Document for Cataloging")

                columns_list = []
                columns_list = df.columns.values.tolist()

                secure_fields = st.multiselect(
                    "Secure Fields Privacy Fields of Dataset", columns_list)
                st.write(secure_fields)
                technical_metadata = extract_technical_metadata(
                    object_name, secure_fields)
                business_metadata = extract_business_metadata(
                    description, database_name, keywords)
                administrative_metadata = extract_administrative_metadata(
                    organization_unit, business_unit, department, data_owner, domain_name)
                semantics_metadata = extract_semantics_metadata(object_name)
                operational_metadata = extract_operational_metadata(source)

                metadata = {
                    "technical_metadata": technical_metadata,
                    "business_metadata": business_metadata,
                    "administrative_metadata": administrative_metadata,
                    "semantics_metadata": semantics_metadata,
                    "operational_metadata": operational_metadata
                }

                # Convert to JSON format
                json_metadata = json.dumps(metadata, indent=4)

                # Display the JSON metadata
                st.subheader("Combined Metadata")
                st.json(json_metadata)

                # Create RDF graph
                graph = Graph()

                # Define namespaces
                metadata = Namespace("metadata#")
                rdfs = RDFS

                # Parse JSON metadata
                metadata_dict = json.loads(json_metadata)
                # Add semantics metadata
                semantics_metadata = metadata["SemanticsMetadata"]
                graph.add((semantics_metadata, RDF.type, metadata["Metadata"]))

                for concept in metadata_dict["semantics_metadata"]["concepts"]:
                    concept_uri = metadata[concept]
                    graph.add((concept_uri, RDF.type, metadata["Concept"]))
                    graph.add((concept_uri, rdfs.label, Literal(concept)))
                    graph.add(
                        (semantics_metadata, metadata["hasConcept"], concept_uri))

                for relationship in metadata_dict["semantics_metadata"]["relationships"]:
                    relationship_uri = metadata[relationship["source"] +
                                                "-" + relationship["target"]]
                    graph.add((relationship_uri, RDF.type,
                               metadata["Relationship"]))
                    graph.add((relationship_uri, rdfs.label,
                               Literal(relationship["relation"])))
                    graph.add(
                        (semantics_metadata, metadata["hasRelationship"], relationship_uri))

                # Convert RDF graph to networkx graph
                # networkx_graph = rdflib_to_networkx_multidigraph(graph)
                # Plot the graph
                networkx_graph = rdflib_to_networkx_multidigraph(graph)

                # Perform spring layout
                pos = nx.spring_layout(networkx_graph)

                # Plot the graph
                plt.figure(figsize=(12, 8))
                nx.draw(networkx_graph, pos, with_labels=True, node_color="skyblue",
                        node_size=2000, font_size=10, edge_color="gray", arrowsize=12, linewidths=1)
                plt.title("RDF Graph")
                plt.axis("off")

                # Save the graph as a PNG image
                image_stream = io.BytesIO()
                plt.savefig(image_stream, format="png")
                image_stream.seek(0)

                # Save the image to a file
                ontology_graph_name = os.path.splitext(
                    object_name)[0] + "_ontology.png"
                with open(ontology_graph_name, "wb") as file:
                    file.write(image_stream.read())
                st.success(
                    f"Ontology Metadata Graph is Generated and saved to {ontology_graph_name}.")

                # Save the JSON metadata to a file
                metadata_file_name = os.path.splitext(
                    object_name)[0] + "_metadata.json"
                with open(metadata_file_name, "w") as file:
                    file.write(json_metadata)
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

                with open(ontology_graph_name, 'rb') as file_data:
                    file_stat = os.stat(ontology_graph_name)
                    client.put_object(
                        'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{database_name}/" + ontology_graph_name, file_data, file_stat.st_size)

                with open(object_name, 'rb') as file_data:
                    file_stat = os.stat(object_name)

                    # client.put_object('sree', file_path_name, file_data,
                    #                  file_stat.st_size,)
                    client.put_object(
                        'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{database_name}/" + object_name, file_data, file_stat.st_size)

                    # print(f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                    st.write(
                        f"my_app render: {(datetime.now() - start_time).total_seconds()}s")
                    st.success(
                        f"DataSet named {object_name}. ingested succesfully to Raw Zone of Data Lake")

                index_metadata(json_metadata, index_name)
                # st.write("Index of Document Created Succesfully")
                st.success("Metadata Indexed to Data Catloging Succesfully")

        if option == 'Postgres SQL':

            host_name = st.text_input("Database Server Name")
            db_user_name = st.text_input("Database User Name")
            db_user_password = st.text_input("Database User Password")
            db_name = st.text_input("Database Name")
            db_port = st.text_input("Database Port Number")
            conn = psycopg2.connect(
                host=host_name, port=db_port, dbname=db_name, user=db_user_name, password=db_user_password)
            cur = conn.cursor()
            list_table_querry = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
            cur.execute(list_table_querry)
            records = cur.fetchall()
            for table_list in records:
                st.write(table_list)
            options = []
            for (tables_post,) in records:
                options.append(tables_post)
            tables_selected = st.multiselect("Tables List", options)
            # mydb.commit()
            # st.table(table_list)
            st.write(tables_selected)
            # st.write(tables_selected[1])
            for tables in tables_selected:
                st.write(tables)
            st.write(
                "Export Tables of Your Choice in to CSV for Ingestion to Data Lake ")
            for tables in tables_selected:
                # sql = "COPY (SELECT * FROM a_table WHERE month=6) TO STDOUT WITH CSV DELIMITER ';'"
                table_export_querry = " COPY ( select * FROM " + \
                    tables + ")" + " TO STDOUT WITH CSV DELIMITER ';' "
                file_path_name = tables + "." + "csv"
                with open(file_path_name, "w") as file:
                    cur.copy_expert(table_export_querry, file)
                client = Minio('172.16.51.28:9000',
                               access_key='minio',
                               secret_key='miniostorage',
                               secure=False)

                with open(file_path_name, 'rb') as file_data:
                    file_stat = os.stat(file_path_name)

                    client.put_object('sree', file_path_name, file_data,
                                      file_stat.st_size,)

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
                    # aws_access_key_id='',
                    # aws_secret_access_key='',
                    aws_access_key_id=aws_access_key,
                    aws_secret_key=aws_secret_key,
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
                    list_object = []
                    for s3_object in my_bucket.objects.all():
                        list_object.append(s3_object.key)
                        object_selected = st.multiselect(
                            "Objects List", list_object)
                        for object in object_selected:
                            filename = s3_object.key
                            my_bucket.download_file(s3_object.key, filename)
                            with open(filename, 'rb') as file_data:
                                file_stat = os.stat(filename)
                                minio_client.put_object(
                                    'sree', filename, file_data, file_stat.st_size)
