# SCBE-AETHERMOORE Web API Docker Image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY symphonic_cipher/ ./symphonic_cipher/
COPY web/ ./web/

# Environment
ENV PORT=8080
ENV DEBUG=false
ENV API_KEYS=demo-key

EXPOSE 8080

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "web.app:app"]
