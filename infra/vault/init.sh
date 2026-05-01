#!/bin/sh
# VayuGrid — Vault initialisation script
# Run once after the Vault dev server has started to seed secrets and policies.
#
# Usage:
#   VAULT_ADDR=http://localhost:8200 VAULT_TOKEN=vayugrid-dev-token sh infra/vault/init.sh

set -e

: "${VAULT_ADDR:=http://localhost:8200}"
: "${VAULT_TOKEN:=vayugrid-dev-token}"

export VAULT_ADDR VAULT_TOKEN

echo "==> Enabling KV v2 secrets engine at path 'vayugrid'…"
vault secrets enable -path=vayugrid kv-v2 2>/dev/null || echo "    (already enabled)"

echo "==> Storing initial AES-GCM encryption key…"
vault kv put vayugrid/encryption-key \
  value="uN_v58Z_V7P-v1A8A-v1_V7P-v1A8A-v1_V7P-v1A="

echo "==> Writing vayugrid-api policy…"
vault policy write vayugrid-api - <<'POLICY'
path "vayugrid/*" {
  capabilities = ["read", "list"]
}
path "vayugrid/encryption-key" {
  capabilities = ["read", "create", "update"]
}
POLICY

echo "==> Vault initialised for VayuGrid."
echo "    VAULT_ADDR  = $VAULT_ADDR"
echo "    Key path    = vayugrid/encryption-key"
