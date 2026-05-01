# HashiCorp Vault — VayuGrid Key Management

VayuGrid uses Vault (self-hosted) to manage the AES-256-GCM encryption key used to
protect personal data stored in the database.

## Quick start (development)

1. Start all services (Vault runs in dev mode via docker-compose):
   ```bash
   docker compose up vault -d
   ```

2. Seed secrets and policies:
   ```bash
   VAULT_ADDR=http://localhost:8200 VAULT_TOKEN=vayugrid-dev-token sh infra/vault/init.sh
   ```

3. The API automatically reads the encryption key from Vault on startup when
   `VAULT_ADDR` and `VAULT_TOKEN` are set. If Vault is unavailable it falls back
   to the `ENCRYPTION_KEY` environment variable.

## Key rotation

Send a POST request to the admin key-rotation endpoint (operator role required):

```bash
curl -X POST http://localhost:8000/api/admin/rotate-key \
  -H "Authorization: Bearer <operator-token>"
```

This generates a new 32-byte random key, stores it in Vault, and returns the new
key value. **Re-encrypt any existing data before deploying the new key to production.**

## Production checklist

- [ ] Switch Vault from dev mode to HA (Raft storage) before production deployment
- [ ] Replace the dev root token with AppRole or Kubernetes auth
- [ ] Enable audit logging: `vault audit enable file file_path=/vault/logs/audit.log`
- [ ] Set up 90-day automated key rotation via a cron job calling the rotate-key endpoint
- [ ] Back up the Vault unseal keys and root token in a secure offline location
