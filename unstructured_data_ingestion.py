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
from datetime import datetime
# File Processing Pkgs
import docx2txt
from PyPDF2 import PdfFileReader
import pdfplumber
from PIL import Image
import os
import json
import magic
import time
import docx
from tika import parser
import spacy
from spacy.lang.en import English
from rake_nltk import Rake
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
import pytesseract
import json
import re
import moviepy.editor as mp
import pytesseract
from nltk import word_tokenize, pos_tag, ne_chunk
import nltk

# Download the 'averaged_perceptron_tagger' data
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Functi

nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer
stop_words = set(stopwords.words("english"))


def extract_technical_metadata(file_path):
    technical_metadata = {}

    # Extract file type
    file_type = magic.from_file(file_path, mime=True)
    technical_metadata['file_type'] = file_type

    # Extract file size
    file_size = os.path.getsize(file_path)
    technical_metadata['file_size'] = file_size

    # Extract creation date
    creation_date = os.path.getctime(file_path)
    technical_metadata['creation_date'] = time.ctime(creation_date)

    # Extract last modified date
    last_modified_date = os.path.getmtime(file_path)
    technical_metadata['last_modified_date'] = time.ctime(last_modified_date)

    return technical_metadata


def extract_business_metadata(text, description, document_name):
    business_metadata = {}

    # Extract keywords
    keywords = extract_keywords(text)
    business_metadata['keywords'] = keywords

    # Extract categories (placeholder)
    business_metadata['categories'] = ['finance', 'technology']
    business_metadata['description'] = description,
    business_metadata['document_name'] = document_name,

    return business_metadata


def extract_keywords(text):
    """
    keywords = []

    doc = nlp(text)
    for token in doc:
        if not token.is_stop and token.is_alpha:
            keywords.append(token.lemma_.lower())
    """
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    return keywords


def extract_operational_metadata(file_path):
    operational_metadata = {}

    # Extract author
    author = 'John Doe'  # Replace with appropriate logic to extract author information
    operational_metadata['author'] = author

    # Extract created date
    created_date = time.ctime(os.path.getctime(file_path))
    operational_metadata['created_date'] = created_date

    # Extract modified date
    modified_date = time.ctime(os.path.getmtime(file_path))
    operational_metadata['modified_date'] = modified_date

    return operational_metadata


def extract_semantics_metadata(file_path):
    semantics_metadata = {}

    # Parse file content using tika parser
    parsed_content = parser.from_file(file_path)
    content = parsed_content['content']

    # Extract context
    semantics_metadata['context'] = content

    # Extract domain (placeholder)
    semantics_metadata['domain'] = 'example_domain'

    # Extract entities (placeholder)
    entities = extract_entities(content)
    semantics_metadata['entities'] = entities

    """# Extract concepts (placeholder)
    concepts = extract_concepts(content)
    semantics_metadata['concepts'] = concepts

    # Extract relationships (placeholder)
    relationships = extract_relationships(content)
    semantics_metadata['relationships'] = relationships
    """
    # Extract topics and their hierarchies
    topics = extract_topics(content)
    simple_topics = extract_simple_topics(content)
    #keywords = extract_keywords(content)
    #semantics_metadata['keywords'] = keywords
    semantics_metadata['simple_topics'] = simple_topics
    semantics_metadata['topics'] = topics

    return semantics_metadata


def extract_entities(text):
    entities = []

    doc = nlp(text)
    for entity in doc.ents:
        if entity.label_ in ['PERSON', 'ORG', 'GPE']:
            entities.append(entity.text)

    return entities


def extract_concepts(text):
    concepts = []
    doc = nlp(text)
    for sentence in doc.sents:
        sentence_concepts = []
        for token in sentence:
            if not token.is_stop and token.is_alpha:
                sentence_concepts.append(token.lemma_.lower())
        if sentence_concepts:
            concepts.append(sentence_concepts)

    return concepts

    # Placeholder logic to extract concepts from the text
    # Replace it with your own logic or use external libraries

    return concepts


def extract_relationships(text):
    relationships = []
    doc = nlp(text)
    for sentence in doc.sents:
        sentence_relationships = []
        for token in sentence:
            if token.dep_ != 'punct':
                sentence_relationships.append(
                    (token.head.lemma_.lower(), token.lemma_.lower()))
        if sentence_relationships:
            relationships.append(sentence_relationships)
    # Placeholder logic to extract relationships from the text
    # Replace it with your own logic or use external libraries

    return relationships


def extract_topics(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Remove stopwords and tokenize the sentences into words
    word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenized_sentences = [word_tokenizer.tokenize(
        sentence.lower()) for sentence in sentences]
    tokenized_sentences = [[word for word in sentence if word not in stop_words]
                           for sentence in tokenized_sentences]

    # Create a document-term matrix
    vectorizer = CountVectorizer(lowercase=False)
    doc_term_matrix = vectorizer.fit_transform(
        [' '.join(sentence) for sentence in tokenized_sentences])

    # Perform topic modeling using Latent Dirichlet Allocation (LDA)
    # Adjust the number of topics as per your requirement
    lda_model = LatentDirichletAllocation(n_components=5)
    lda_model.fit(doc_term_matrix)

    # Extract the dominant topic for each sentence
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for i, topic_prob in enumerate(lda_model.transform(doc_term_matrix)):
        topic = {'topic_id': i, 'topic_keywords': []}
        # Adjust the number of keywords per topic
        topic_keywords = [feature_names[j] for j in topic_prob.argsort()[-5:]]
        topic['topic_keywords'] = topic_keywords
        topics.append(topic)

    return topics


def extract_simple_topics(text):
    doc = nlp(text)
    simple_topics = []

    for sent in doc.sents:
        if len(sent) > 2:
            simple_topics.append(sent.text)

    return simple_topics


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


def extract_text_from_image(image_path):
    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
    return text

# Function for keyword extraction using RAKE
def extract_keywords_with_rake_from_image(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords

# Function for keyword extraction using NLP (spaCy)
def extract_keywords_with_nlp_from_image(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return keywords

# Function for named entity recognition (NER) using spaCy
def extract_image_named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entities = {
        "people": [entity.text for entity in doc.ents if entity.label_ == "PERSON"],
        "organizations": [entity.text for entity in doc.ents if entity.label_ == "ORG"],
        "places": [entity.text for entity in doc.ents if entity.label_ == "GPE"],
    }
    return entities

# Function to extract technical metadata
def extract_image_technical_metadata(image_path):
    with Image.open(image_path) as img:
        technical_metadata = {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "number_of_pixels": img.size[0] * img.size[1],
        }
    return technical_metadata

def extract_image_business_metadata(text, description, data_owner):
    # Placeholder for business metadata extraction
    # For demonstration purposes, let's assume some example business metadata
    business_metadata = {
        "description": description,
        "data_owner": data_owner,
        "keywords": extract_keywords_with_rake_from_image(text),
    }
    return business_metadata


# Function to extract operational metadata
def extract_image_operational_metadata(image_path):
    # Placeholder for operational metadata extraction
    # For demonstration purposes, let's assume some example operational metadata

    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get the file creation time
            creation_time = os.path.getctime(image_path)

            # Get the file modification time
            modification_time = os.path.getmtime(image_path)

            # Get the file access time
            access_time = os.path.getatime(image_path)

            # Get other image properties
            format_name = img.format
            mode = img.mode
            size = img.size
            dpi = img.info.get('dpi', None)  # Image resolution (dots per inch)

    except Exception as e:
        # Handle any exceptions that might occur while opening the image or fetching properties
        print(f"Error: {e}")
        return None

    operational_metadata = {
        "creation_time": creation_time,
        "modification_time": modification_time,
        "access_time": access_time,
        "format": format_name,
        "mode": mode,
        "size": size,
        "dpi": dpi,
    }
    return operational_metadata


def extract_image_semantics_metadata(text):
    semantics_metadata = {
        "keywords_nlp": extract_keywords_with_nlp_from_image(text),
        "entities": extract_image_named_entities(text),
    }
    return semantics_metadata


def extract_image_administrative_metadata(organization_unit, business_unit, department, data_owner, domain_name):
    # Construct the business metadata dictionary from user inputs
    administrative_metadata = {
        "organization_unit": organization_unit,
        "business_unit": business_unit,
        "department": department,
        "data_owner": data_owner,
        "domain_name": domain_name
    }
    return administrative_metadata



def extract_video_technical_metadata(video_file):
    video = mp.VideoFileClip(video_file)
    metadata = {
        "duration": video.duration,
        "fps": video.fps,
        "size": video.size,
        "created": os.path.getctime(video_file),
        "modified": os.path.getmtime(video_file),
        "accessed": os.path.getatime(video_file),
        "owner": os.getlogin()
    }
    video.close()
    return metadata

# Function to extract business metadata from video file
def extract_video_business_metadata(captions_text, description):
    # You can implement your logic here to extract business metadata.
    # This could involve reading from a database or external source.
    # For demonstration purposes, let's return some example data.
    doc = nlp(captions_text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha and token.pos_ != "PRON"]
    business_metadata = {
        "keywords": keywords,
        "description": description,
        "owner": "John Doe",
        "location": "New York"
    }
    return business_metadata


# Function to extract operational metadata from video file
def extract_video_operational_metadata(video_file, data_owner):
    video = mp.VideoFileClip(video_file)
    metadata = {
        "created": os.path.getctime(video_file),
        "modified": os.path.getmtime(video_file),
        "accessed": os.path.getatime(video_file),
        "data_owner": data_owner, 
        "owner": os.getlogin() or os.environ.get("USERNAME")
    }
    video.close()
    return metadata

# Function to extract captions from video file using OCR
def extract_captions_text(video_file, frame_sampling_rate=10):
    video = mp.VideoFileClip(video_file)
    captions_text = ""
    frame_number = 0
    for frame in video.iter_frames(fps=video.fps, dtype="uint8"):
        if frame_number % frame_sampling_rate == 0:
            frame_text = pytesseract.image_to_string(frame)
            captions_text += frame_text + " "
        frame_number += 1
    video.close()
    return captions_text

# Function to extract semantic metadata from video captions
def extract_video_semantic_metadata(captions_text):
    entities = []
    keywords = []
    topics = []
    tokens = word_tokenize(captions_text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)

    for entity in named_entities:
        if hasattr(entity, 'label') and entity.label:
            entities.append(" ".join([token[0] for token in entity.leaves()]))

    # Additional logic to extract keywords and topics from the text.
    # You may use TF-IDF, TextRank, LDA, or other techniques depending on your requirements.

    return {
        "entities": entities,
        #"keywords": keywords,
        "topics": topics
    }



def extract_video_administrative_metadata(organization_unit, business_unit, department, data_owner, domain_name):
    # Construct the business metadata dictionary from user inputs
    administrative_metadata = {
        "organization_unit": organization_unit,
        "business_unit": business_unit,
        "department": department,
        "data_owner": data_owner,
        "domain_name": domain_name
    }
    return administrative_metadata


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


def load_image(image_file):
    img = Image.open(image_file)
    return img


def app():
    image = Image.open('msis.jpeg')

    st.image(image, width=100)

    st.title('Welcome to Data Ingestion of Un-Structured Data Types')

    st.sidebar.title('Un-Structured DataTypes')
    option = st.sidebar.selectbox(
        'select subjects', ('Text', 'Videos', 'Images'))

    if option == 'Text':

        st.sidebar.title('Text Upload Types')
        option = st.sidebar.selectbox(
            'select text_upload_types', ('Local File Browse', 'URL Link', 'Object Storage Types'))
        if option == 'Local File Browse':
            st.subheader("DocumentFiles")
            client = Minio('172.16.51.28:9000',
                           access_key='minio',
                           secret_key='miniostorage',
                           secure=False)

            docx_files = st.file_uploader("Upload Document", type=[
                "pdf", "docx", "txt"], accept_multiple_files=True)
            if docx_files is not None:
                for docx_file in docx_files:
                    file_details = {"filename": docx_file.name, "filetype": docx_file.type,
                                    "filesize": docx_file.size}
                    st.write(file_details['filename'])

                    if docx_file.type == "text/plain":
                        raw_text = str(docx_file.read(), "utf-8")
                        st.text(raw_text)
                        st.write(file_details['filename'])
                        #tempDir = "/home/msis/sreephd_data_ingestion_service/dis_version_1_ingestion_hetrogenous_types/."
                        start_time = datetime.now()
                        tempDir = os.getcwd()
                        with open(os.path.join(tempDir, file_details['filename']), "wb") as f:
                            f.write(docx_file.getbuffer())
                        organization_unit = st.text_input(
                            "Organization Unit Name")
                        business_unit = st.text_input("Business Unit Name")
                        department = st.text_input("Department Name")
                        data_owner = st.text_input("Data Owner Name")
                        domain_name = st.text_input("Data Domain/Context Name")
                        document_name = st.text_input("Document Name of Data")
                        description = st.text_input(
                            "Description About DataSet")
                        source = st.text_input("Source of DataSet")
                        #keywords = st.text_input("Keywords of Dataset")
                        index_name = st.text_input(
                            "Index Name of Document for Cataloging")
                        object_name = file_details['filename']

                        technical_metadata = extract_technical_metadata(
                            object_name)
                        business_metadata = extract_business_metadata(
                            object_name, description, document_name)
                        administrative_metadata = extract_administrative_metadata(
                            organization_unit, business_unit, department, data_owner, domain_name)
                        semantics_metadata = extract_semantics_metadata(
                            object_name)
                        operational_metadata = extract_operational_metadata(
                            object_name)

                        metadata = {
                            "technical_metadata": technical_metadata,
                            "business_metadata": business_metadata,
                            "administrative_metadata": administrative_metadata,
                            "semantics_metadata": semantics_metadata,
                            "operational_metadata": operational_metadata
                        }
                        # Convert to JSON format
                        json_metadata = json.dumps(metadata, indent=4)
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
                                'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{document_name}/" + metadata_file_name, file_data, file_stat.st_size)

                        with open(object_name, 'rb') as file_data:
                            file_stat = os.stat(object_name)

                            client.put_object(
                                'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{document_name}/" + object_name, file_data, file_stat.st_size)

                        st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                        st.success(
                            f"DataSet named {object_name}. ingested succesfully to Raw Zone of Data Lake")

                        index_metadata(json_metadata, index_name)
                        st.success(
                            "Metadata Indexed to Data Catloging Succesfully")

                        st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                    elif docx_file.type == "application/pdf":
                        try:
                            with pdfplumber.open(docx_file) as pdf:
                                pages = pdf.pages[0]
                                st.write(pages.extract_text())
                                start_time = datetime.now()
                                tempDir = os.getcwd()
                                #tempDir = "/home/msis/sreephd_data_ingestion_service/."
                                with open(os.path.join(tempDir, file_details['filename']), "wb") as f:
                                    f.write(docx_file.getbuffer())
                                organization_unit = st.text_input(
                                    "Organization Unit Name")
                                business_unit = st.text_input(
                                    "Business Unit Name")
                                department = st.text_input("Department Name")
                                data_owner = st.text_input("Data Owner Name")
                                domain_name = st.text_input(
                                    "Data Domain/Context Name")
                                document_name = st.text_input(
                                    "Document Name of Data")
                                description = st.text_input(
                                    "Description About DataSet")
                                source = st.text_input("Source of DataSet")
                                #keywords = st.text_input("Keywords of Dataset")
                                index_name = st.text_input(
                                    "Index Name of Document for Cataloging")
                                object_name = file_details['filename']

                                technical_metadata = extract_technical_metadata(
                                    object_name)
                                business_metadata = extract_business_metadata(
                                    object_name, description, document_name)
                                administrative_metadata = extract_administrative_metadata(
                                    organization_unit, business_unit, department, data_owner, domain_name)
                                semantics_metadata = extract_semantics_metadata(
                                    object_name)
                                operational_metadata = extract_operational_metadata(
                                    object_name)

                                metadata = {
                                    "technical_metadata": technical_metadata,
                                    "business_metadata": business_metadata,
                                    "administrative_metadata": administrative_metadata,
                                    "semantics_metadata": semantics_metadata,
                                    "operational_metadata": operational_metadata
                                }
                                # Convert to JSON format
                                json_metadata = json.dumps(metadata, indent=4)
                                # Save the JSON metadata to a file
                                metadata_file_name = os.path.splitext(
                                    object_name)[0] + "_metadata.json"
                                with open(metadata_file_name, "w") as file:
                                    file.write(json_metadata)
                                st.success(
                                    f"Metadata saved to {metadata_file_name}.")

                                Catalog = "Catalogs"
                                Data = "Data"
                                client = Minio('172.16.51.28:9000',
                                               access_key='minio',
                                               secret_key='miniostorage',
                                               secure=False)
                                with open(metadata_file_name, 'rb') as file_data:
                                    file_stat = os.stat(metadata_file_name)
                                    client.put_object(
                                        'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{document_name}/" + metadata_file_name, file_data, file_stat.st_size)

                                with open(object_name, 'rb') as file_data:
                                    file_stat = os.stat(object_name)

                                    client.put_object(
                                        'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{document_name}/" + object_name, file_data, file_stat.st_size)

                                st.write(
                                    f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                                st.success(
                                    f"DataSet named {object_name}. ingested succesfully to Raw Zone of Data Lake")

                                index_metadata(json_metadata, index_name)
                                st.success(
                                    "Metadata Indexed to Data Catloging Succesfully")

                                st.write(
                                    f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                        except:
                            st.write("None")
                    else:
                        raw_text = docx2txt.process(docx_file)
                        st.write(raw_text)
                        #tempDir = "/home/msis/sreephd_data_ingestion_service/."
                        start_time = datetime.now()
                        tempDir = os.getcwd()
                        with open(os.path.join(tempDir, file_details['filename']), "wb") as f:
                            f.write(docx_file.getbuffer())
                        organization_unit = st.text_input(
                            "Organization Unit Name")
                        business_unit = st.text_input("Business Unit Name")
                        department = st.text_input("Department Name")
                        data_owner = st.text_input("Data Owner Name")
                        domain_name = st.text_input("Data Domain/Context Name")
                        document_name = st.text_input("Document Name of Data")
                        description = st.text_input(
                            "Description About DataSet")
                        source = st.text_input("Source of DataSet")
                        #keywords = st.text_input("Keywords of Dataset")
                        index_name = st.text_input(
                            "Index Name of Document for Cataloging")
                        object_name = file_details['filename']
                        technical_metadata = extract_technical_metadata(
                            object_name)
                        business_metadata = extract_business_metadata(
                            object_name, description, document_name)
                        administrative_metadata = extract_administrative_metadata(
                            organization_unit, business_unit, department, data_owner, domain_name)
                        semantics_metadata = extract_semantics_metadata(
                            object_name)
                        operational_metadata = extract_operational_metadata(
                            object_name)

                        metadata = {
                            "technical_metadata": technical_metadata,
                            "business_metadata": business_metadata,
                            "administrative_metadata": administrative_metadata,
                            "semantics_metadata": semantics_metadata,
                            "operational_metadata": operational_metadata
                        }
                        # Convert to JSON format
                        json_metadata = json.dumps(metadata, indent=4)
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
                                'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{document_name}/" + metadata_file_name, file_data, file_stat.st_size)

                        with open(object_name, 'rb') as file_data:
                            file_stat = os.stat(object_name)
                            client.put_object(
                                'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{document_name}/" + object_name, file_data, file_stat.st_size)
                        st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                        st.success(
                            f"DataSet named {object_name}. ingested succesfully to Raw Zone of Data Lake")

                        index_metadata(json_metadata, index_name)
                        st.success(
                            "Metadata Indexed to Data Catloging Succesfully")

                        st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

    if option == 'Images':
        st.sidebar.title('Image Upload Types')
        option = st.sidebar.selectbox(
        'select image_upload_types', ('Local File Browse', 'URL Link', 'Object Storage Types'))
        if option == 'Local File Browse':
            st.subheader("Image Dataset")
            client = Minio('172.16.51.28:9000', access_key='minio',
                           secret_key='miniostorage', secure=False)
            st.subheader("Image")
            uploaded_files = st.file_uploader(
                "Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

            if uploaded_files is not None:
                for image_file in uploaded_files:
                    file_details = {"filename": image_file.name, "filetype": image_file.type,
                                    "filesize": image_file.size}
                    st.write(file_details)
                    st.image(load_image(image_file), width=250)
                     #fileDir = "/home/msis/sreephd_data_ingestion_service/."
                    start_time = datetime.now()
                    fileDir = os.getcwd()
                    with open(os.path.join(fileDir, image_file.name), "wb") as f:
                        f.write((image_file).getbuffer())

                    organization_unit = st.text_input(
                            "Organization Unit Name")
                    business_unit = st.text_input("Business Unit Name")
                    department = st.text_input("Department Name")
                    data_owner = st.text_input("Data Owner Name")
                    domain_name = st.text_input("Data Domain/Context Name")
                    image_name = st.text_input("Document Name of Data")
                    description = st.text_input(
                            "Description About DataSet")
                    source = st.text_input("Source of DataSet")
                    #keywords = st.text_input("Keywords of Dataset")
                    index_name = st.text_input(
                            "Index Name of Document for Cataloging")
                    object_name = image_file.name
                    extracted_text = extract_text_from_image(object_name)
                    # Step 2: Extract technical metadata
                    technical_metadata = extract_image_technical_metadata(object_name)

                    # Step 3: Extract business metadata
                    business_metadata = extract_image_business_metadata(extracted_text, description, data_owner)

                    # Step 4: Extract operational metadata
                    operational_metadata = extract_image_operational_metadata(object_name)

                    # Step 5: Extract semantics metadata
                    semantics_metadata = extract_image_semantics_metadata(extracted_text)
                    administrative_metadata = extract_image_administrative_metadata(
                            organization_unit, business_unit, department, data_owner, image_name)

                    # Combine all metadata into a single dictionary
                    metadata = {
                    "technical": technical_metadata,
                    "business": business_metadata,
                    "operational": operational_metadata,
                    "administrative": administrative_metadata,
                    "semantics": semantics_metadata,
                    }
                    json_metadata = json.dumps(metadata, indent=4)
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
                                'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{image_name}/" + metadata_file_name, file_data, file_stat.st_size)

                    with open(object_name, 'rb') as file_data:
                        file_stat = os.stat(object_name)
                        client.put_object(
                                'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{image_name}/" + object_name, file_data, file_stat.st_size)
                    st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                    st.success(
                            f"DataSet named {object_name}. ingested succesfully to Raw Zone of Data Lake")

                    index_metadata(json_metadata, index_name)
                    st.success(
                            "Metadata Indexed to Data Catloging Succesfully")

                    st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")




                    with open(file_details['filename'], 'rb') as file_data:
                        file_stat = os.stat(file_details['filename'])
                        client.put_object(
                                'sree', file_details['filename'], file_data, file_stat.st_size)

                    st.write(
                                f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

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
                    start_time = datetime.now()
                    fileDir = os.getcwd()
                    #fileDir = "/home/msis/sreephd_data_ingestion_service/."
                    with open(os.path.join(fileDir, video_file.name), "wb") as f:
                        f.write((video_file).getbuffer())

                    
                    organization_unit = st.text_input(
                            "Organization Unit Name")
                    business_unit = st.text_input("Business Unit Name")
                    department = st.text_input("Department Name")
                    data_owner = st.text_input("Data Owner Name")
                    domain_name = st.text_input("Data Domain/Context Name")
                    video_name = st.text_input("Document Name of Data")
                    description = st.text_input(
                            "Description About DataSet")
                    source = st.text_input("Source of DataSet")
                    #keywords = st.text_input("Keywords of Dataset")
                    index_name = st.text_input(
                            "Index Name of Document for Cataloging")
                    object_name = video_file.name

                    technical_metadata = extract_video_technical_metadata(object_name)
                    operational_metadata = extract_video_operational_metadata(object_name, data_owner)
                    captions_text = extract_captions_text(object_name)

                    business_metadata = extract_video_business_metadata(captions_text, description)
                    semantic_metadata = extract_video_semantic_metadata(captions_text)
                    administrative_metadata = extract_video_administrative_metadata(
                            organization_unit, business_unit, department, data_owner, video_name)

                    metadata = {
                    "technical_metadata": technical_metadata,
                    "business_metadata": business_metadata,
                    "operational_metadata": operational_metadata,
                    "administrative_metadata": administrative_metadata,
                    "semantic_metadata": semantic_metadata
                        }

                    json_metadata = json.dumps(metadata, indent=4)
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
                                'metadata-test', f"{Catalog}/{organization_unit}/{business_unit}/{department}/{domain_name}/{video_name}/" + metadata_file_name, file_data, file_stat.st_size)

                    with open(object_name, 'rb') as file_data:
                        file_stat = os.stat(object_name)
                        client.put_object(
                                'metadata-test', f"{Data}/{organization_unit}/{business_unit}/{department}/{domain_name}/{video_name}/" + object_name, file_data, file_stat.st_size)
                    st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

                    st.success(
                            f"DataSet named {object_name}. ingested succesfully to Raw Zone of Data Lake")

                    index_metadata(json_metadata, index_name)
                    st.success(
                            "Metadata Indexed to Data Catloging Succesfully")

                    st.write(
                            f"my_app render: {(datetime.now() - start_time).total_seconds()}s")

