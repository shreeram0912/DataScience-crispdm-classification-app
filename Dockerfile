# Base Image:
FROM python:3.13.3

# Setting working directory which includes all required files to run the app in a single directory named as:
WORKDIR /app
    
# Copy requirements file from CRISPDM folder
COPY requirements.txt .
    
# Installing dependencies: To ensure that the application can run consistently across different environments.
RUN pip install --no-cache-dir -r requirements.txt
    
# Copying application code from all files
COPY . .
    
# EXPOSE port for Streamlit to run on localhost 8501
EXPOSE 8501
    
# EXPOSE port for MLflow to run on localhost 5000
EXPOSE 5000
    
# Run the application using command line:
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 & streamlit run finapp.py"]
    