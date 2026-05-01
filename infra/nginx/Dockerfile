FROM nginx:1.27-alpine
# Copy config; TLS certs are mounted at runtime via docker-compose volume
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 8443
