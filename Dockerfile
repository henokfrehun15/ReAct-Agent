FROM python:3.12-slim

ENV PYTHONPATH=/app/src
WORKDIR /app

RUN pip install --no-cache-dir openai python-dotenv

COPY src/ ./src/
COPY data/ ./data/

RUN useradd -m -s /bin/bash agentuser && chown -R agentuser:agentuser /app
USER agentuser

ENTRYPOINT ["python", "-m", "agent.cli"]