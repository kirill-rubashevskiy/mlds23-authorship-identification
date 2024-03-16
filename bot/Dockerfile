FROM python:3.11-slim as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY pyproject.toml poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --with bot


FROM python:3.11-slim

WORKDIR /code

COPY /conf ./conf
COPY /bot ./bot
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

ENV PYTHONPATH=/code

RUN apt-get update && apt-get full-upgrade -y && apt-get install gcc -y
RUN pip install wheel setuptools pip --upgrade && pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python3", "bot/main.py"]