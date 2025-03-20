# data_lake_service_home_page.py
import streamlit as st
import webbrowser

def app():
    # Set the page configuration to exclude the navigation sidebar
    #st.set_page_config(layout="wide")

    # Add your welcome message in a div or header
    st.title("Welcome to Data Lake Services of MAHE")
    st.header("Welcome to the Data Lake Services")
    st.write("This is the default landing page for Data Lake Services of MAHE.")

    # Add your images and styling here
    image_url = "data_lake.png"
    st.image(image_url, caption="Data Lake Image", use_column_width=True)

    # Add buttons to navigate to other services
    st.write("Select a service to access:")

    # Create a row layout for the buttons
    col1, col2, col3 = st.columns(3)

    structured_clicked = col1.button("Structured Data Ingestion")
    streaming_clicked = col2.button("Streaming Data Ingestion")
    unstructured_clicked = col3.button("Unstructured Data Ingestion")

    # Create a row layout for the buttons
    col4, col5 = st.columns(2)

    nosql_clicked = col4.button("NoSQL Data Ingestion")
    medical_clicked = col5.button("Medical HealthCare Data")

    # Display descriptions based on button clicks
    if not any([structured_clicked, streaming_clicked, unstructured_clicked,
                nosql_clicked, medical_clicked]):
        st.subheader("Service Descriptions:")
        st.write("**Structured Data Ingestion**:")
        st.write("This service allows you to ingest structured data.")
        st.write("**Streaming Data Ingestion**:")
        st.write("This service allows you to ingest streaming data.")
        st.write("**Unstructured Data Ingestion**:")
        st.write("This service allows you to ingest unstructured data.")
        st.write("**NoSQL Data Ingestion**:")
        st.write("This service allows you to ingest NoSQL data.")
        st.write("**Medical HealthCare Data**:")
        st.write("This service allows you to ingest medical and healthcare data.")

    if structured_clicked:
        open_url_in_new_tab("http://172.16.51.38:8050")

    if streaming_clicked:
        open_url_in_new_tab("http://172.16.51.38:8051")

    if unstructured_clicked:
        open_url_in_new_tab("http://172.16.51.38:8052")

    if nosql_clicked:
        open_url_in_new_tab("http://172.16.51.38:8053")

    if medical_clicked:
        open_url_in_new_tab("http://172.16.51.38:8054")

def open_url_in_new_tab(url):
    new_tab = 2
    webbrowser.open(url, new=new_tab)

