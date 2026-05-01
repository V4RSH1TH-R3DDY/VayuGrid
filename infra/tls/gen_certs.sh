#!/bin/bash
# VayuGrid — mTLS Certificate Generator
#
# Generates a self-signed CA, a server certificate for the nginx reverse proxy,
# and a sample Vayu-Node client certificate.
#
# Run once before deploying the nginx mTLS service.
# Requires: openssl
#
# NOTE: Make this script executable before use:
#   chmod +x infra/tls/gen_certs.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CA_DIR="$DIR/ca"
SERVER_DIR="$DIR/server"
CLIENT_DIR="$DIR/client"
mkdir -p "$CA_DIR" "$SERVER_DIR" "$CLIENT_DIR"

# ── 1. Root CA ────────────────────────────────────────────────────────────────
echo "==> Generating Root CA…"
openssl genrsa -out "$CA_DIR/ca.key" 4096
openssl req -new -x509 -days 3650 \
  -key "$CA_DIR/ca.key" \
  -out "$CA_DIR/ca.crt" \
  -subj "/C=IN/ST=Karnataka/L=Bangalore/O=VayuGrid/OU=CA/CN=VayuGrid Root CA"

# ── 2. Server certificate (for nginx) ────────────────────────────────────────
echo "==> Generating server certificate…"
openssl genrsa -out "$SERVER_DIR/server.key" 2048
openssl req -new \
  -key "$SERVER_DIR/server.key" \
  -out "$SERVER_DIR/server.csr" \
  -subj "/C=IN/ST=Karnataka/L=Bangalore/O=VayuGrid/OU=API/CN=vayugrid-api"
openssl x509 -req -days 365 \
  -in  "$SERVER_DIR/server.csr" \
  -CA  "$CA_DIR/ca.crt" \
  -CAkey "$CA_DIR/ca.key" \
  -CAcreateserial \
  -out "$SERVER_DIR/server.crt"

# ── 3. Sample Vayu-Node client certificate ───────────────────────────────────
echo "==> Generating sample client certificate (node-001)…"
openssl genrsa -out "$CLIENT_DIR/node-001.key" 2048
openssl req -new \
  -key "$CLIENT_DIR/node-001.key" \
  -out "$CLIENT_DIR/node-001.csr" \
  -subj "/C=IN/ST=Karnataka/L=Bangalore/O=VayuGrid/OU=VayuNode/CN=node-001"
openssl x509 -req -days 365 \
  -in  "$CLIENT_DIR/node-001.csr" \
  -CA  "$CA_DIR/ca.crt" \
  -CAkey "$CA_DIR/ca.key" \
  -CAcreateserial \
  -out "$CLIENT_DIR/node-001.crt"

echo ""
echo "✅ Certificates generated in $DIR/"
echo "   CA cert     : $CA_DIR/ca.crt"
echo "   Server cert : $SERVER_DIR/server.crt"
echo "   Server key  : $SERVER_DIR/server.key"
echo "   Client cert : $CLIENT_DIR/node-001.crt"
echo "   Client key  : $CLIENT_DIR/node-001.key"
echo ""
echo "Mount $DIR into the nginx container at /etc/nginx/certs (see docker-compose.yml)."
