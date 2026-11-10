# Deploying so a reviewer can try it in seconds

Goal: an employer clicks one link, sees a working app, and can't accidentally (or on purpose) run up your API bill.

## How the safeguards work (read this first)

| Layer | What it does | Where you set it |
|---|---|---|
| **Search-only for everyone** | Without the access code, visitors can still search the documents and see retrieved passages. Free, needs no key. | automatic |
| **Access code** | Live Claude answers + the agent unlock only with your code. You give it to recruiters in your email/resume. | `APP_ACCESS_CODE` |
| **Daily cap** | Hard limit on live answers per day across all visitors (default 100). | `DAILY_QUERY_LIMIT` |
| **Spend limit** | The real backstop. In the Anthropic console, set a low monthly limit (e.g. $5-10) on the key you use for the demo. | console.anthropic.com |

Use a **separate API key just for the demo** so you can revoke it without touching anything else.

---

## Option A: Streamlit Community Cloud (free, a link you can put on a resume)

1. **Push to GitHub.** Create a repo named `rag-agent-eval`, then:
   ```bash
   git init
   git add .
   git status        # CHECK: .env must NOT be listed
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<you>/rag-agent-eval.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app**, choose your repo, branch `main`, main file `app.py`.
3. Open **Advanced settings > Secrets** and paste (see `.streamlit/secrets.toml.example`):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-demo-key"
   APP_ACCESS_CODE = "pick-a-code"
   DAILY_QUERY_LIMIT = "100"
   ```
4. Deploy. You get a URL like `https://<something>.streamlit.app`.
5. If the build is slow, crashes, or runs out of memory (PyTorch is heavy for a free host), set `EMBEDDER = "hashing"` in Secrets and delete the `sentence-transformers` line from `requirements.txt`. Retrieval gets weaker (keyword-based instead of semantic), so say so in your README if you do.
6. Free apps go to sleep after inactivity. **Open your link yourself right before an interview or application push** so it's awake.

## Option B: Docker (one command, runs the same everywhere)

Good if the reviewer is technical, and good for you to say "it's containerized."

```bash
cp .env.example .env      # add ANTHROPIC_API_KEY (and optionally APP_ACCESS_CODE)
docker compose up --build
# open http://localhost:8501
```

## Option C: One-command local run (no Docker)

```bash
./run.sh        # Mac/Linux (first time: chmod +x run.sh)
run.bat         # Windows (double-click)
```
Creates the virtual environment, installs everything, creates `.env`, and starts the app.

## Option D: Hugging Face Spaces (free, popular with AI hiring managers)

Create a new Space with the **Docker** SDK, push this repo to it, and add the same variables as Space **Secrets**. The included `Dockerfile` listens on 8501, so add `app_port: 8501` to the Space's README front matter.

---

## What to send an employer

> **Live demo:** https://your-app.streamlit.app (search works instantly; the access code for live Claude answers is `<code>`)
> **Code + evaluation results:** https://github.com/<you>/rag-agent-eval

Tip: run `python eval.py --mode all` once locally and **commit `eval_results/latest_summary.md`**. The app shows it in the "About this project" panel, so reviewers see real numbers without running anything.

## Status of these instructions

The access gate, daily cap, example buttons, tests and local launchers were exercised during development. The Docker image, the GitHub Actions workflow, `run.sh`/`run.bat` end-to-end, and the hosting steps above were **written but not run** (no Docker daemon or hosting accounts were available), so expect to fix small things the first time, and tell me what error you see.
