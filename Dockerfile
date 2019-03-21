FROM pytorch/pytorch
MAINTAINER Fabio Graetz
EXPOSE 8000
RUN apt-get update && apt-get install -y apache2 \
    apache2-dev \
    emacs \
    python3 \
    python3-pip\
    python3-dev \
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/lists/*

ADD deviseApi/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
ADD deviseApi /app

WORKDIR /var/www/deviseApi/
COPY ./deviseApi.wsgi /var/www/deviseApi/deviseApi.wsgi
COPY ./deviseApi /var/www/deviseApi/

RUN /opt/conda/bin/mod_wsgi-express install-module
RUN mod_wsgi-express setup-server deviseApi.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND