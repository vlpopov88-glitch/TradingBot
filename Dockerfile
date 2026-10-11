# Runs the bot 24/7. State (paper account, journal, logs) lives in the /app/state volume.
# If Docker Hub rate-limits you, build with the same official image from AWS's mirror:
#   docker build --build-arg BASE=public.ecr.aws/docker/library/python:3.12-slim .
ARG BASE=python:3.12-slim
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
# anthropic: Claude news scoring is used only when ANTHROPIC_API_KEY is set.
RUN pip install --no-cache-dir -r requirements.txt "anthropic>=1.0"

COPY tradingbot ./tradingbot
COPY config.yaml .

RUN useradd --create-home --uid 1000 bot && mkdir -p /app/state && chown bot:bot /app/state
USER bot
VOLUME ["/app/state"]

# Unhealthy if no trading cycle has completed for 20 minutes.
HEALTHCHECK --interval=5m --timeout=10s --start-period=10m \
  CMD python -c "import os,sys,time; p='/app/state/heartbeat'; sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 1200 else 1)"

ENTRYPOINT ["python", "-m", "tradingbot"]
CMD ["run"]
