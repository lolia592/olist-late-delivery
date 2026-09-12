FROM python:3.14-slim

# Matches the host's absolute project path exactly, so MLflow's
# absolute artifact paths (recorded on the host) resolve correctly
# inside the container too — no path translation needed.
WORKDIR /home/lolia/olist-eda

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

COPY app app
COPY src src
COPY config config

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
