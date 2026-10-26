# Web app: running it and deploying it

## Run it on your own machine

You need Python 3, Node 18 or newer and a running Postgres. From the project root:

    pip3 install -r requirements.txt
    python3 src/get_data.py
    brew install postgresql@16 node
    brew services start postgresql@16
    cd web && npm install && npm run build && cd ..
    python3 -m uvicorn server.main:app --port 8000

Open http://127.0.0.1:8000. The first start trains the models if `models/demo.joblib` is missing, which takes about a minute, and creates the `nids` database and its tables.

If `python3` is not the interpreter your packages were installed into (Homebrew can change which one it points to), run both `pip install` and `uvicorn` with the same full path, for example `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pip ...`.

`DATABASE_URL` defaults to `postgresql+psycopg://localhost/nids`, which is your OS user with no password, the way Homebrew sets Postgres up. To use another database, set `DATABASE_URL` before starting the API.

## Deploy it

The repo has a `Dockerfile` that builds the frontend, downloads the dataset and trains the models at build time, so the server starts in a second or two. It installs the CPU-only build of XGBoost to keep the image small. The running app uses about 400 MB of RAM, so pick a plan with at least 1 GB. Free tiers with 512 MB are too tight.

The app reads `PORT` and `DATABASE_URL`, and accepts both the `postgres://` and `postgresql://` URL forms. Set `ALLOW_RESET=0` on a public deployment so visitors cannot clear the stored history.

### Vercel with a Neon database

`Dockerfile.vercel` is the same image set up for Vercel, which runs it as a function on port 80. Vercel and Neon both have free plans that need no card.

1. Create a free database at neon.com and copy its connection string.
2. On vercel.com, choose Add New, then Project, and import this GitHub repo. Leave the framework preset on Other.
3. Before deploying, add an environment variable `DATABASE_URL` with the Neon connection string.
4. Deploy. The first build takes a few minutes because it trains the models.
5. Open the project URL. The overview page should load and say "Backend connected".

Vercel's container support is new, so if the build fails, send the last lines of the build log.

### Other hosts

Any service that builds a `Dockerfile` and offers Postgres works the same way, for example Google Cloud Run with a Neon database, or Railway when it is open for signups. Use `Dockerfile`, set `DATABASE_URL`, and the host's `PORT` is picked up automatically.
