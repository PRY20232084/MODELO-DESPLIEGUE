FROM python:3.9-slim
ENV PYTHONBUFFERED True
ENV TF_ENABLE_ONEDNN_OPTS 0
WORKDIR /app
COPY . .

#RUN pip install --no-cache-dir --upgrade pip
#RUN pip install -r requirements.txt
##RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#EXPOSE 5000
EXPOSE 8081
CMD ["python","app.py"]
#CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app