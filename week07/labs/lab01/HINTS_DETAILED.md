# Hints (Detailed Tier)

This tier shows the working core of each task with line by line commentary
explaining why each piece is there. It deliberately withholds the final assembly:
completing a task still means reading, understanding, and putting the pieces
together in your own cell. If you want to reason your way there with lighter
nudges instead, close this file and open `HINTS.md`. Pick one tier per task, not
both; reading both wastes your lab time.

---

## Task 1: The Dockerfile

The core instructions, annotated. Assemble them in the contract's order inside the
`DOCKERFILE` string.

```dockerfile
FROM python:3.13-slim
```
Pinned tag, matching the cohort's Python 3.13. `:latest` would make every rebuild a
surprise; a pinned tag makes upgrades a decision, not an accident.

```dockerfile
WORKDIR /app
```
Every later relative path and the module import root now hang off `/app`.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
Dependencies first, alone. Docker caches layers top down and invalidates from the
first changed layer onward, so isolating the rarely changing dependency list means
a code edit never re-runs pip. The `--no-cache-dir` flag keeps pip's download cache
out of the layer, which is pure image bloat otherwise.

```dockerfile
COPY app/ ./app/
```
Code after dependencies. This is the layer that changes constantly, so it goes as
late as possible.

```dockerfile
RUN useradd -m appuser && mkdir -p /cache && chown -R appuser /app /cache
```
Three actions chained: create an unprivileged user with a home directory, create
the cache directory, and hand both directories to that user. The `/cache` part
looks pointless today; in Part 6 a named volume mounts there, and an empty named
volume inherits the ownership of the image path it lands on. Without this, the
volume mounts root-owned and the non-root service cannot write to it.

```dockerfile
USER appuser
```
Everything above this line ran as root, which is what pip and useradd need.
Everything from here on, including the final process, runs unprivileged.

```dockerfile
ENV BACKEND_MODE=offline
ENV MODEL=gemma4
```
Safe defaults baked in. These are configuration, not secrets, so `ENV` is
appropriate; secrets never go in an image.

The remaining two instructions are yours: document the port the service listens
on, and start the module with the exec-form `CMD` from the contract. Exec form
(the JSON array) matters because it makes your process PID 1 with no shell in
between, so stop signals reach it directly.

---

## Task 2: The .dockerignore

```
data/
```
The dataset is mounted at runtime in Task 6. Baking data into an image couples a
code artifact to a data snapshot, and it ships every byte to the daemon on every
build.

```
out/
```
Runtime output. It does not exist at build time conceptually, even when the
directory is sitting there.

```
__pycache__/
*.pyc
```
Compiled Python cache. Harmless in the image but pure waste in the context, and it
can differ per machine, which poisons layer cache hits.

```
.git/
```
Repository history is often the largest thing in a project directory and the image
never needs it.

Round the file out yourself with the good-practice entries the contract mentions:
local virtual environments, env files, and large model file patterns.

---

## Task 3: The build command

```
docker build -t cordwell/review-summarizer:v1
```
The verb and the tag. `docker build` routes to BuildKit, and `-t` names the result
so every later command can refer to it instead of an image hash. One argument is
still missing from this line: the build context. The contract says which directory,
and `run_shell` already puts you inside it.

---

## Task 4: The run command

The flags, annotated, in no particular order:

```
-d
```
Detached. Without it the cell blocks forever on the server's foreground output.

```
-p 8000:8000
```
Host port, colon, container port. The service listens on 8000 inside; this maps
your Mac's 8000 onto it.

```
--name review-summarizer
```
A stable handle. The logs cell, the cleanup cell, and the checks all refer to the
container by this name.

```
-e BACKEND_MODE=offline
```
Explicit config at run time, even though the image default is the same. The habit
is the lesson: the run command documents the configuration.

Assemble these between `docker run` and the image tag, image last.

---

## Task 5: The runtime override

The command is your Task 4 answer with one addition:

```
-e MODEL=cordwell-eval
```
The image bakes `MODEL=gemma4` as a default; `-e` outranks `ENV` at container
start. The service reads the environment once at startup, which is why the demo
replaces the container instead of poking the old one. Nothing was rebuilt: same
image ID, different behavior, and that difference lives entirely in the run
command.

---

## Task 6: The batch command

The pieces, annotated:

```
--rm
```
A one-off job should not leave a stopped container behind. `--rm` removes it the
moment the process exits.

```
-v "$(pwd)/data/reviews.csv:/data/reviews.csv:ro"
```
Bind mount, three segments: absolute host path, container path, and the read-only
flag. `$(pwd)` expands to the lab directory because that is where `run_shell`
executes. The `:ro` means even a buggy job cannot corrupt the source data.

```
-v "$(pwd)/out:/out"
```
The output mount, read-write by default. Files the job writes to `/out` land in
`cordwell-lab/out` on your Mac and survive the container's removal.

```
python -m app.batch --in /data/reviews.csv --out /out
```
Everything after the image tag replaces the image's `CMD`. Note the paths are the
container-side paths, because that is where this command runs.

Assemble: `docker run`, the cleanup flag, both mounts, the backend env flag, the
image tag, then the command override.

---

## Task 7: The host-backend command

The three environment flags for the Ollama variant, annotated:

```
-e BACKEND_MODE=ollama
```
Selects the live code path in `backend.py`. An unknown value raises a clear error
instead of silently falling back; explicit beats implicit.

```
-e OLLAMA_BASE_URL=http://host.docker.internal:11434
```
The whole lesson in one flag. `localhost:11434` inside the container is the
container itself; `host.docker.internal` is Docker Desktop's stable DNS name for
your Mac, where Ollama actually listens.

```
-e MODEL=gemma4
```
The model tag the host server has pulled. Confirm the exact tag on your machine.

For LM Studio, both change together: mode `lmstudio` and base URL
`http://host.docker.internal:1234/v1`.

The rest of the command is plumbing you have already written twice: detached,
`--rm`, a fresh `--name`, a port mapping that avoids the busy 8000 (the contract
names the port), and the image tag.

---

## Task 8: The Compose file

The six TODO fragments, annotated:

**TODO 1**, one more environment entry on the summarizer:
```yaml
      - CACHE_DIR=/cache
```
The service only caches when this variable is set. Pointing it at `/cache` aims it
exactly where the volume mounts.

**TODO 2**, the mount on the summarizer:
```yaml
    volumes:
      - summary_cache:/cache
```
Name, colon, container path. Because the left side is a name rather than a host
path, this is a named volume, not a bind mount, and Docker manages its storage.

**TODO 3**, the healthcheck timing, siblings of `test`:
```yaml
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s
```
Poll every 5 seconds, give each probe 3 seconds, allow 5 consecutive failures
before unhealthy, and ignore failures during the first 10 seconds while the
service boots.

**TODO 4**, the gateway's published port:
```yaml
    ports:
      - "5005:8000"
```
Quoted, host first. 5005 because macOS AirPlay owns 5000.

**TODO 5**, the readiness gate on the gateway:
```yaml
    depends_on:
      summarizer:
        condition: service_healthy
```
The mapping form, because only it can carry a condition. The plain list form
orders container starts and nothing more.

**TODO 6** is yours to write: a top-level section declaring the volume by name.
Two lines, starting at column zero, and the second line ends with a colon and
nothing after it. The progressive tier shows it verbatim if you want confirmation.

---

## Stretch 1: Multi-stage build

The two stage boundary, annotated:

```dockerfile
FROM python:3.13-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt
```
Stage one exists only to produce `/venv`. Naming it with `AS builder` lets the next
stage refer to it.

```dockerfile
FROM python:3.13-slim
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"
```
Stage two starts clean and copies exactly one artifact across. The `PATH` edit
makes the venv interpreter the `python` that `CMD` finds, which is how the
dependencies travel without pip ever running in the final image.

The rest of the final stage is your Task 1 Dockerfile from `COPY app/` onward,
including the `/cache` setup, the user switch, the defaults, and the same `CMD`.
Build with the `-f Dockerfile.multistage` flag and the `v2` tag from the notebook,
then compare `docker images` output for the two tags.

---

## Stretch 2: Image healthcheck

The one new instruction, annotated:

```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
```
Exec form again, and the probe is python with urllib because the slim image ships
no curl. Inside this container `localhost` is correct: the probe runs in the same
network namespace as the service. It goes before `CMD` in an otherwise unchanged
copy of your Task 1 Dockerfile.

The build, run, wait, and inspect sequence is spelled out in the notebook's
stretch description. For the bonus question, think about which definition is
closer to the deployment: the image travels everywhere, the Compose file describes
one specific deployment.

---

## Stretch 3: Scaling and the port gotcha

No code core to show; the two commands are given in the notebook. The reasoning
scaffold: list what each service claims from the host machine. The summarizer
claims nothing (no ports), so replicas only need distinct names on the private
network, which Compose handles. The gateway claims host port 5005, and a host port
is a singleton resource. When the second replica asks for it, the bind fails. Both
standard fixes work by making the claim non-singleton: a port range gives each
replica its own port, and a reverse proxy moves the single published port to a new
front service so the gateway replicas stop claiming host ports at all. One extra
observation worth making in this stack: all summarizer replicas mount the same
named volume, so a summary cached by one replica is a cache hit on the others.
