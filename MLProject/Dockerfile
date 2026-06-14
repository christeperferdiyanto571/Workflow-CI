FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install mlflow scikit-learn pandas numpy matplotlib

CMD ["python", "modelling.py"]
