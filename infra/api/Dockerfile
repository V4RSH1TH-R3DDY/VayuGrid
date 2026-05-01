FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY . .

WORKDIR /app/api
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
