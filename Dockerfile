# 1. Base image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements trước để tận dụng cache
COPY requirements.txt .

# 4. Cài dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy code và model
COPY build_api.py .
COPY catboost_info/catboost_v2.cbm ./catboost_info/catboost_v2.cbm

# 6. Khai báo port
EXPOSE 8000

# 7. Khởi động API
CMD ["uvicorn", "build_api:app", "--host", "0.0.0.0", "--port", "8000"]