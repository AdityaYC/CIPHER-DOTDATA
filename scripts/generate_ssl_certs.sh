#!/usr/bin/env bash
# Generate self-signed SSL certs so the server can run over HTTPS.
# Safari on iPhone requires HTTPS for camera (getUserMedia). First visit: tap Advanced → Continue.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSL_DIR="$(dirname "$SCRIPT_DIR")/ssl"
mkdir -p "$SSL_DIR"
cd "$SSL_DIR"
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'
echo "Created $SSL_DIR/cert.pem and $SSL_DIR/key.pem"
echo "Run the server; it will use HTTPS if these files exist. On iPhone open https://YOUR_MAC_IP:8000/phone and accept the certificate."
