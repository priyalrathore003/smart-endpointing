# Rules for working in this repo

`SPEC.md` is the source of truth. Work stage by stage (SPEC section 8) and stop after each stage.

- Stage explicit paths only; never `git add -A` or `git add .`.
- Before every commit, scan the staged diff for secrets (`sk-`, `AKIA`, `aws_secret`,
  private keys, API tokens). If anything is found, stop.
- Read API keys only from environment variables. Never print, log or commit them.
- Never read `.env` (or `.env.*`) files.
- Never force-push.
- Ask before anything that costs money (LLM API calls, AWS), with a cost estimate first.
