FROM python:3.10-slim

WORKDIR /training-pipeline

COPY . .

RUN pip install -r requirements.txt

CMD ["python","train.py"]