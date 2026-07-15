# Security policy

Report suspected vulnerabilities privately to the repository owner. Do not open a public issue containing credentials, conversation data, exploit payloads, or production identifiers.

## Deployment checklist

1. Set `ENVIRONMENT=production`.
2. Generate a long random `API_AUTH_TOKEN` and send it as `X-API-Key` from the bot.
3. Configure `SUPABASE_SERVICE_ROLE_KEY`; never ship it to a browser or Discord client.
4. Apply `db/migrations/002_harden_api_data.sql` before switching the API to service-role access.
5. Restrict `TRUSTED_HOSTS`, keep CORS empty unless a browser client is required, and allow only exact origins.
6. Keep `INGEST_ALLOWED_HOSTS` limited to trusted attachment CDNs.
7. Run the quality and security commands documented in `AGENTS.md`.

Rotate a secret immediately if it appears in logs, commits, screenshots, support tickets, or chat history.
