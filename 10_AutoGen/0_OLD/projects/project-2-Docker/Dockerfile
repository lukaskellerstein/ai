FROM python:3.11-slim

WORKDIR /app

COPY ./OAI_CONFIG_LIST.json ./
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./script.py ./

ENTRYPOINT ["tail", "-f", "/dev/null"]