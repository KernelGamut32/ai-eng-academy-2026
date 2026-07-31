# Local W&B Server Setup (Docker)

The lab defaults to the `offline` backend for Weights & Biases, which needs nothing. Set up
the local server if you want the full UI, comparison views, parallel
coordinates, and the artifact lineage graph, with no data leaving your
machine. Same pattern as Pinecone Local earlier this week: a vendor
server, self-hosted in Docker, free for local use.

Do this before class, not during. The image pull is over a gigabyte.

## 1. Weights & Biases (W&B) Local (Docker)

If not already executed, download **wandb-local-latest.tar** from the **images** folder at <https://gamuttechnologysvcs-my.sharepoint.com/:f:/p/asanders/IgD_SIVCz8YJQYh7BL3DUy4ZAVgU8-9SO8Lo3boIy-wwV8g?e=D4X53b>. From the location where **wandb-local-latest.tar** has been stored, run `docker load -i wandb-local-latest.tar` to load the image to local cache

```bash
docker run -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
```

The container maps port 8080 and stores data in a persistent
Docker volume named `wandb`, so your runs survive container restarts.

Note the CLI's own help says this command is for local testing only; a
production deployment of W&B Server uses their Kubernetes operator. Local
testing is exactly what we are doing.

## 2. First-time license and account

Open <http://localhost:8080>. On first boot, create the local admin
account, then paste a free license at
<http://localhost:8080/system-admin> when prompted. The license for local
personal use is free from the W&B deployment page; the UI links to it.
This step exists once per machine.

## 3. Point your client at it

```bash
wandb login --host=http://localhost:8080
```

Paste the API key the local UI shows you. After this, the lab handles
the rest: `WANDB_LAB_BACKEND=local` makes `lab_support` set
`WANDB_BASE_URL=http://localhost:8080` before wandb is imported.

## 4. Verify, and what to look at

After Part D, open the project, select the three runs, and open the
comparison view: it shows the same table your Part E DataFrame shows.
After Part F, open the `failure-analysis-baseline` run and sort the
`eval/per_query` table by `answer_relevancy` ascending; the two
unanswerable questions surface immediately. After Part C on this
backend, the artifact pages for `cordwell-adapter` and `cordwell-eval`
show the lineage graph with producer and consumer runs.

## 6. Teardown and restart

```bash
docker stop wandb-local
```

Data persists in the `wandb` volume. `wandb server start` or the
`docker run` line brings it back with everything intact. To wipe
completely:

- `docker rm <container-ID> -f` to remove
- `docker volume rm wandb` after removal

## Fallbacks

| Symptom | Fix |
|---|---|
| Port 8080 already in use | Stop the other service, or run the container with `-p 8081:8080` and set `WANDB_BASE_URL=http://localhost:8081` manually before Jupyter |
| Apple Silicon image warning | The `wandb/local` image runs under emulation on some tags; slower startup is normal. If it fails to boot, use `offline` for the lab; the notebook builds every comparison programmatically, so nothing is lost but the UI views |
| Login loop or license issues | `offline` backend; nothing in the lab requires the server |
| Forgot to log in before Jupyter | Log in from a terminal, then restart the Jupyter kernel |

The universal fallback is always `WANDB_LAB_BACKEND=offline`. Every
check passes there; the runs sync to any server later with
`wandb sync ./wandb/offline-run-<name>`.
