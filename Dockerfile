FROM python:3.9-slim
WORKDIR /app
COPY . /app/
RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install libmagic-dev libtesseract-dev  -y
RUN apt-get install python3-pip -y
RUN pip3 install -r requirnments.txt
RUN python -m spacy download en_core_web_sm 
EXPOSE 8502 
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
