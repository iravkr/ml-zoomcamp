FROM python:3.10-slim

RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["predict.py", "model.bin", "./"]
EXPOSE 4041
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]