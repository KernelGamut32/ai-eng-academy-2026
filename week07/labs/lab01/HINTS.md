# Hints (Progressive Tier)

Three escalating levels per task. Read level 1 first, and only go deeper if you are
still stuck after trying. If you would rather see the working core of a task with
line by line commentary, close this file and open `HINTS_DETAILED.md` instead. Pick
one tier per task, not both; reading both wastes your lab time.

---

## Task 1: The Dockerfile

**Level 1.** The contract in the notebook is an ordered recipe. Write one Dockerfile
instruction per numbered item, top to bottom. The check cell tells you exactly which
requirement is missing, so iterate against it: write, run the check, fix, repeat.

**Level 2.** The skeleton is: `FROM`, `WORKDIR`, `COPY` the requirements file, `RUN`
pip install, `COPY` the app package, one `RUN` for the user and directory setup,
`USER`, two `ENV` lines, `EXPOSE`, `CMD`. The order of the two `COPY` steps relative
to the pip install is the whole caching lesson: dependencies change rarely, code
changes constantly, and Docker reuses layers top down until the first change. The
user setup line chains three shell commands with `&&`.

**Level 3.** The two lines students most often fumble, in context:

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
...
RUN useradd -m appuser && mkdir -p /cache && chown -R appuser /app /cache
```

The first must come after `COPY requirements.txt .` and before `COPY app/ ./app/`.
The second must come before `USER appuser`, which must come before `CMD`. The `CMD`
is the exec form with three list items: python, the `-m` flag, and `app.serve`.

---

## Task 2: The .dockerignore

**Level 1.** Think about what is in `cordwell-lab` that the image does not need:
the dataset is mounted at runtime, outputs are produced at runtime, and caches and
repository history are never wanted. One exclusion per line.

**Level 2.** Four entries are required by the check: the `data` directory, the `out`
directory, `__pycache__`, and `.git`. Directory entries conventionally end with a
slash. Adding `.venv`, `.env`, and large model file patterns is good practice.

**Level 3.** A minimal passing file is four lines: `data/`, `out/`, `__pycache__/`,
`.git/`. Compare yours against the check output to see which entry it thinks is
missing.

---

## Task 3: The build command

**Level 1.** One command, three pieces: the build verb, a tag flag, and a build
context. The notebook told you the exact tag to use.

**Level 2.** The shape is `docker build -t TAG CONTEXT`. The context is the current
directory because `run_shell` executes inside `cordwell-lab`, and the current
directory is spelled with a single dot.

**Level 3.** `docker build -t cordwell/review-summarizer:v1 .` and the dot is not
optional: it is the argument that says what to send to the daemon.

---

## Task 4: The run command

**Level 1.** Every bullet in the contract maps to exactly one flag. Assemble them in
any order between `docker run` and the image tag.

**Level 2.** You need: `-d`, `-p` with host port and container port separated by a
colon, `--name`, one `-e` for the backend mode, and then the image. The `-p` order
trips people: host first, container second.

**Level 3.** The port flag is `-p 8000:8000` and the env flag is
`-e BACKEND_MODE=offline`. The image tag goes last. If the check passes but the
health call fails, read `docker logs review-summarizer` before changing anything.

---

## Task 5: The runtime override

**Level 1.** This is Task 4 again with one difference. The provided line already
removed the old container, so the name is free. What single flag changes?

**Level 2.** Keep every flag from Task 4 and add a second `-e` that sets `MODEL`.
The image is unchanged and unrebuilt; that is the point of the exercise.

**Level 3.** Add `-e MODEL=cordwell-eval` alongside `-e BACKEND_MODE=offline`. The
health endpoint reads the model name from the environment at startup, so the fresh
container reports the new value.

---

## Task 6: The batch command

**Level 1.** A one-off job wants three things a service does not: it should remove
itself when done, it needs its input and output paths mapped to the host, and it
overrides the image's default command with the batch entrypoint.

**Level 2.** Two `-v` flags. Each is `host_path:container_path` and the input one
gets a third segment, `:ro`. Host paths must be absolute; use `"$(pwd)/..."` so the
shell expands them from the lab directory. The command override goes after the
image tag, verbatim from the contract.

**Level 3.** The two mounts, in context:

```
-v "$(pwd)/data/reviews.csv:/data/reviews.csv:ro" -v "$(pwd)/out:/out"
```

Combine with `--rm`, `-e BACKEND_MODE=offline`, the image tag, and then
`python -m app.batch --in /data/reviews.csv --out /out`.

---

## Task 7: The host-backend command

**Level 1.** The trap named in the notebook is the answer: inside the container,
`localhost` is not your Mac. What hostname does Docker Desktop give you for the
host, and which port does your chosen backend listen on?

**Level 2.** Three `-e` flags: the backend mode, the matching base URL through
`host.docker.internal`, and the model tag. Publish `-p 8001:8000` so it does not
collide with the Task 5 container, and add `--rm`, `-d`, and a distinct `--name`.

**Level 3.** For Ollama the base URL flag is
`-e OLLAMA_BASE_URL=http://host.docker.internal:11434` and the mode flag is
`-e BACKEND_MODE=ollama`. For LM Studio, swap both: mode `lmstudio`, URL
`http://host.docker.internal:1234/v1`. The check rejects any `localhost` in the
URL, and that rejection is the lesson.

---

## Task 8: The Compose file

**Level 1.** Each TODO comment in the skeleton is one small YAML fragment. Work top
to bottom and keep the indentation of the surrounding lines; YAML errors here are
almost always indentation errors.

**Level 2.** TODO 1 is one more list item under `environment`. TODO 2 introduces a
`volumes:` key on the summarizer with one list item in `name:path` form. TODO 3 is
four timing keys at the same level as `test`. TODO 4 is a `ports:` key with one
quoted `host:container` item. TODO 5 uses the mapping form of `depends_on`, not the
list form, because only the mapping form can carry a condition. TODO 6 is a
top-level section with the volume name and nothing under it.

**Level 3.** The two fragments students most often get wrong, in context:

```yaml
    depends_on:
      summarizer:
        condition: service_healthy
```

```yaml
volumes:
  summary_cache:
```

The first lives under the gateway service, indented like its siblings. The second
starts at column zero, outside `services:`, and the trailing colon with nothing
after it is correct: it declares the volume with default settings.

---

## Stretch goals

Progressive hints stop here by design. The stretch goals are specified fully in the
notebook, and `HINTS_DETAILED.md` covers each one at working-core depth if you want
the heavier path.
