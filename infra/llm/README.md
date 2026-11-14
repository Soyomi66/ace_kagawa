# Local LLM for Tokkio

Tokkio reads the LLM through an OpenAI-compatible endpoint (`TOKKIO_LLM_BASE_URL`, port 8011 on
this host). Each compose file here serves one model with vLLM on GPU0; only one runs at a time,
because they share the port and the GPU memory budget (0.55 x 48 GB).

| File | Model | Status |
| --- | --- | --- |
| `docker-compose.gemma4-26b-a4b.yml` | Gemma 4 26B A4B, 4-bit AWQ (`gemma-4-26b-a4b-it`) | Production since 2026-11-13 (`docs/audits/llm-gemma4-26b-a4b-2026-11-13.md`) |
| `docker-compose.gemma4-12b.yml` | Gemma 4 12B, FP8 (`gemma-4-12b-it`) | Fallback (`docs/audits/llm-candidates-2026-10-28.md`) |
| `docker-compose.swallow8b.yml` | Llama-3.1-Swallow-8B | Previous production, kept for rollback |

`Dockerfile.vllm-cu129` builds the vLLM image the Gemma files use (v0.32.0-cu129 without torchcodec).

## Start and check

`manage_tokkio.sh start` does not start the LLM. Start it first:

```bash
docker start ace-vllm-gemma4-26b-a4b        # or: docker compose -f infra/llm/docker-compose.gemma4-26b-a4b.yml up -d
python3 infra/llm/check_llm_endpoint.py --base-url http://127.0.0.1:8011/v1 --model gemma-4-26b-a4b-it
```

`docker` needs containerd, which runs only while Tokkio's Kubernetes is up.

## Switching models

Stop the running container, start the other compose file, and set `TOKKIO_LLM_MODEL` in
`infra/tokkio/.env` to the served name. The served name must contain `gemma-4` for
`infra/tokkio/tokkio_llm.py` to send the evaluated sampling with thinking off. The full procedure
is in `infra/tokkio/README.md` (ローカル LLM を Gemma 4 12B（FP8）に切り替える).
