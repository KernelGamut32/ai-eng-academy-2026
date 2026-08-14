# Week 7 Module 01 Quiz: ML Pipelines, Automation & API Design (Docker)

**Instructions.** Five multiple choice questions. Choose the single best answer for each. Questions 1 through 3 show code from the Cordwell services covered in this module; read the code before answering. No notes, no terminal. Budget about 10 minutes.

---

## Question 1

A teammate rewrites the `review-summarizer` Dockerfile as follows. It builds successfully and the container runs correctly.

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY app/ ./app/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "app.serve"]
```

The teammate then fixes a one line bug in `app/serve.py` and rebuilds. Compared to the layer ordering taught in this module, what is the practical consequence of this Dockerfile?

**A.** The rebuild fails, because `requirements.txt` is copied after the application code and pip cannot resolve the dependency list.

**B.** The rebuild succeeds, but the cache is invalidated at the `COPY app/ ./app/` layer, so `pip install` re-runs on every code edit even though the dependencies did not change.

**C.** The rebuild succeeds with no practical difference, because BuildKit compares file contents per instruction and only re-runs the layers whose inputs changed.

**D.** The rebuild succeeds, but the final image is larger, because the application code is committed into two separate layers.

---

## Question 2

The `sku-search` Compose file wires the API to Pinecone Local like this:

```yaml
services:
  api:
    build: ./api
    ports:
      - "5000:5000"
    depends_on:
      pinecone:
        condition: service_started
    environment:
      - PINECONE_API_KEY=pclocal
      - PINECONE_HOST=http://pinecone.local:5080
  pinecone:
    image: ghcr.io/pinecone-io/pinecone-local:latest
    platform: linux/amd64
    environment:
      PORT: 5080
      PINECONE_HOST: pinecone.local
    networks:
      default:
        aliases:
          - pinecone.local
```

On a cold `docker compose up`, the `api` service sometimes crashes immediately with `connection refused` against `pinecone.local:5080`. Running `docker compose up` again a few seconds later works. What is the root cause?

**A.** The `depends_on` gate only waits for the Pinecone container to start, not for the service inside it to accept connections, so the API can race ahead of an emulated container that is still initializing.

**B.** The `pinecone.local` network alias is not registered until the first DNS query against it resolves, so the first lookup from the API always fails.

**C.** Port 5080 is not listed under a `ports:` key for the `pinecone` service, so it is unreachable until Docker lazily publishes it on first use.

**D.** The `PINECONE_API_KEY` value is a placeholder, so the first connection is rejected during authentication and succeeds only after the client retries.

---

## Question 3

Ollama is running natively on the instructor's Mac and answers correctly when called from a host terminal. The instructor then starts the containerized service:

```bash
docker run --rm -p 8000:8000 \
  -e BACKEND_MODE=ollama \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  -e MODEL=gemma4 \
  review-summarizer:v1
```

Every request through the container fails with `connection refused` on port 11434. What single change fixes it?

**A.** Add `-p 11434:11434` to the `docker run` command so the Ollama port is published alongside port 8000.

**B.** Add `-e OLLAMA_HOST=0.0.0.0` to the `docker run` command so the container process listens on all interfaces.

**C.** Change `OLLAMA_BASE_URL` to `http://host.docker.internal:11434`, because inside the container `localhost` refers to the container itself, not the Mac.

**D.** Change `BACKEND_MODE` to `lmstudio`, because the `ollama` mode is only valid when the model server also runs inside a container.

---

## Question 4

An engineer builds `review-summarizer` on a cohort Mac with `docker build -t registry.example.com/cordwell/review-summarizer:v1 .`, pushes it, and the amd64 cloud host fails to start it with an exec format error. The team wants **one tag** that the amd64 cloud host can run **and** that cohort Macs can pull and run natively, without anyone passing a platform flag at pull time. Which command achieves that?

**A.** `docker buildx build --platform linux/amd64 -t registry.example.com/cordwell/review-summarizer:v1 --load .` followed by `docker push` of the same tag.

**B.** `docker buildx build --platform linux/amd64,linux/arm64 -t registry.example.com/cordwell/review-summarizer:v1 --load .` so both binaries land in the local daemon before pushing.

**C.** `docker buildx build --platform linux/amd64,linux/arm64 -t registry.example.com/cordwell/review-summarizer:v1 --push .` so the registry holds a manifest that resolves to the right binary on each host.

**D.** `docker tag review-summarizer:v1 registry.example.com/cordwell/review-summarizer:v1` followed by `docker push`, because retagging normalizes the image to the registry's default platform.

---

## Question 5

In a single stage Dockerfile, a teammate handles the Pinecone API key like this: a `COPY` instruction brings `pinecone_api_key.txt` into the image, a later `RUN` instruction uses it, and the same `RUN` finishes with `rm` on the file. The final container has no such file on its filesystem. According to this module, what is the security status of that key?

**A.** The key is safe, because deleting the file inside the image means no running container can ever read it.

**B.** The key is safe as long as the deletion happens in the same `RUN` instruction that used it, because install and cleanup in one layer leaves nothing behind.

**C.** The key is exposed only to processes running as root inside the container, so adding a `USER appuser` instruction after the deletion closes the gap.

**D.** The key is retrievable from the image history, because the `COPY` instruction committed it into a layer that later instructions cannot remove; the correct pattern is a BuildKit secret mount.

---

*End of quiz.*
