FROM python:3.13-slim

WORKDIR /app

COPY sherlock/pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY sherlock/sherlock/ sherlock/
COPY instruments/ instruments/

ENV SHERLOCK_INSTRUMENTS_DIR=/app/instruments

EXPOSE 8082

CMD ["uvicorn", "sherlock.main:app", "--host", "0.0.0.0", "--port", "8082"]
