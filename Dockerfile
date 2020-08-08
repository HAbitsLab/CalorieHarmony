FROM python:3.7
WORKDIR /
COPY . ./

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN chmod a+x run.sh

CMD ["./run.sh"]