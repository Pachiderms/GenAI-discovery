FROM nginx:alpine

COPY . /usr/share/nginx/html
RUN rm -f /etc/nginx/conf.d/default.conf
COPY app.conf /etc/nginx/conf.d/app.conf

RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/ssl 

RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
	-out /etc/nginx/ssl/inception.crt \
	-keyout /etc/nginx/ssl/inception.key \
	-subj "/C=FR/ST=Paris/L=Paris/O=42/OU=1337/CN=genai-discovery.local.fr"

EXPOSE 80 443
