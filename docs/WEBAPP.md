# Web app: running it and deploying it

## Run it on your own machine

You need Python 3, Node 18 or newer and a running Postgres. From the project root:

    pip3 install -r requirements.txt
    python3 src/get_data.py
    brew install postgresql@16 node
    brew services start postgresql@16
    cd web && npm install && npm run build && cd ..
    python3 -m uvicorn api.main:app --port 8000

Open http://127.0.0.1:8000. The first start trains the models if `models/demo.joblib` is missing, which takes about a minute, and creates the `nids` database and its tables.

If `python3` is not the interpreter your packages were installed into (Homebrew can change which one it points to), run both `pip install` and `uvicorn` with the same full path, for example `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pip ...`.

`DATABASE_URL` defaults to `postgresql+psycopg://localhost/nids`, which is your OS user with no password, the way Homebrew sets Postgres up. To use another database, set `DATABASE_URL` before starting the API.

## Deploy it

The repo has a `Dockerfile` that builds the frontend, downloads the dataset and trains the models at build time, so the server starts in a second or two. The running app uses about 400 MB of RAM, so pick a plan with at least 1 GB. Free tiers with 512 MB are too tight.

These steps are for Railway. The button names may differ slightly from what you see.

1. Push the repo to GitHub, with the `Dockerfile` in the root.
2. On railway.com, create a new project and choose "Deploy from GitHub repo". Pick this repo. Railway finds the `Dockerfile` by itself. The first build takes a few minutes because it trains the models.
3. In the same project, add a Postgres database ("New", then "Database", then "PostgreSQL").
4. Open the web service, go to Variables, and add `DATABASE_URL` with the value `${{Postgres.DATABASE_URL}}` (the reference to the database you just added). Add `ALLOW_RESET` with the value `0` so visitors cannot clear the stored history.
5. Under Settings, then Networking, generate a public domain. Open it. The overview page should load and say "Backend connected".

The API creates its tables on the first start, so there is nothing to migrate.

Other hosts work the same way: any service that builds a `Dockerfile`, offers Postgres, and sets a `PORT` variable. The app reads `PORT` and `DATABASE_URL`, and accepts both the `postgres://` and `postgresql://` URL forms.
