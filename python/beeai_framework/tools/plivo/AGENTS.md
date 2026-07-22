# Plivo tools — integration notes

Plivo tools for the BeeAI framework: outbound SMS and outbound voice calls.

## Tools
- `PlivoSendMessage` — sends an SMS via `POST https://api.plivo.com/v1/Account/{auth_id}/Message/`. Success is HTTP `202` (message *queued*, not yet delivered).
- `PlivoMakeCall` — places an outbound call via `POST https://api.plivo.com/v1/Account/{auth_id}/Call/`. Success is HTTP `201`. On answer, Plivo fetches `answer_url`, which must return Plivo answer XML (e.g. a `<Speak>` element).

## Configuration
Credentials are read from the environment, never hardcoded:
- `PLIVO_AUTH_ID` — Plivo Auth ID (also the account segment of the REST path).
- `PLIVO_AUTH_TOKEN` — Plivo Auth Token.
- `PLIVO_SRC` — the default sender. For **SMS** this is the `src`: a Plivo number, a short code, or (where the destination country allows) an alphanumeric sender ID. For **calls** this is the `from` caller ID, which must be a voice-capable Plivo number in E.164 — alphanumeric sender IDs are not valid caller IDs.

Auth is HTTP Basic (`auth_id:auth_token`). Numbers are E.164 (with `+`).

Get credentials from the [Plivo console](https://cx.plivo.com/?utm_source=github&utm_medium=oss&utm_campaign=beeai-framework); see the [Plivo API docs](https://www.plivo.com/docs/) for the Message and Call endpoints. This is a REST integration (no audio streaming).

## License / contribution
Apache-2.0. Commits are DCO `Signed-off-by`.
