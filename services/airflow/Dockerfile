FROM apache/airflow:3.0.6

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*
USER airflow


