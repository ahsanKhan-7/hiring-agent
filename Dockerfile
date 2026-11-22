FROM python:3.15-slim
WORKDIR /app
COPY pyproject.toml ./
COPY hiring_agent ./hiring_agent
RUN pip install --no-cache-dir .
ENV CHECKPOINT_DB=/data/checkpoints.db
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "hiring_agent.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
