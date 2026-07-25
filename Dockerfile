FROM python:3.10-slim

# Install system dependencies needed for OpenCV, PyTorch, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from the subfolder
COPY Fetal-Ultrasound-Analysis-ad1c3cfa329daf49eaff1f1c12b625e995067d78/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy subfolder contents into /app
COPY Fetal-Ultrasound-Analysis-ad1c3cfa329daf49eaff1f1c12b625e995067d78/ .

# Ensure env variables are set to bind to 0.0.0.0 and port 7860
ENV FETAL_PORT=7860
ENV FETAL_HOST=0.0.0.0

# Expose port 7860
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]
