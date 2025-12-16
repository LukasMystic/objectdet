# ==================================
# Stage 1: Build Frontend (React/Vite)
# ==================================
# CHANGED: Updated from node:18-alpine to node:20-alpine to support Vite 6+
FROM node:20-alpine as build-frontend

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy source code
COPY frontend/ ./

# CHANGED: Set the API URL to your specific Hugging Face Space domain
# This prevents the App.jsx from falling back to http://127.0.0.1:8000
ENV VITE_API_URL="https://bembeng123-objectdet.hf.space"
RUN npm run build

# ==================================
# Stage 2: Setup Backend (FastAPI)
# ==================================
FROM python:3.9-slim

WORKDIR /app

# CHANGED: Replaced 'libgl1-mesa-glx' with 'libgl1' for newer Debian versions
# ADDED: ca-certificates to ensure SSL/TLS connections to MongoDB work correctly
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend

# Copy built frontend assets from Stage 1
COPY --from=build-frontend /app/frontend/dist ./backend/static

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860
EXPOSE 7860

# Set working directory to backend
WORKDIR /app/backend

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]