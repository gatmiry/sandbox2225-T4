#FROM python:3.8

#WORKDIR .

#COPY testfile2.py .

#CMD ["python", "./testfile2.py"]

FROM python:3.8
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
#RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#USER user
#COPY --link 