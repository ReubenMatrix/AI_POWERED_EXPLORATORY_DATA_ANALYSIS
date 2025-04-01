# Use a base image with Python and Streamlit
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install smolagents

# Copy the Streamlit app
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
