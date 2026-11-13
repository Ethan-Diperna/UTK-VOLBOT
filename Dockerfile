# Container image for the catalog assistant web app.
#
#   docker build -t catalog-assistant .
#   docker run -p 7860:7860 -e ANTHROPIC_API_KEY=... catalog-assistant
#
# The API key is never baked into the image: pass it at run time (on a hosting
# platform, as a secret). The search indexes and both search models are built
# into the image, so the first question does not wait on a download.
#
# Memory: the running app measured about 820 MB (2026-11-13, both models
# loaded). A host with 512 MB is too small; use 1 GB or more.

FROM python:3.11-slim

# Run as a normal user, not root. User id 1000 is also what Hugging Face
# Spaces requires for Docker Spaces.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# The CPU-only build of PyTorch. Without this line pip installs the default
# Linux build, which bundles GPU libraries and adds several GB to the image.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, so editing app code does not reinstall them.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Build the vector and keyword indexes from the committed catalog text, then
# run one search. That downloads the embedding and reranker models into the
# image and fails the build if the indexes do not work.
RUN python -m src.index \
 && python -c "from src import config as cfg; from src.retrieve import Retriever; hits = Retriever.from_disk().search('prerequisites for COSC 360', mode=cfg.RETRIEVAL_MODE, k=cfg.TOP_K); assert hits, 'search returned nothing'; print('index check:', hits[0]['title'])"

# The models are in the image; do not contact Hugging Face at run time.
ENV HF_HUB_OFFLINE=1

# A hosting platform puts its own proxy in front of the container, so every
# request reaches uvicorn from the proxy's address. --proxy-headers makes
# uvicorn read the visitor's address from X-Forwarded-For instead, which the
# per-client rate limit needs. "*" trusts that header from any sender, which
# is only right when the container cannot be reached except through the
# platform's proxy. A visitor can still fake the header to dodge the
# per-client limit; the daily cap (API_DAILY_REQUEST_CAP) is what bounds cost.
ENV FORWARDED_ALLOW_IPS="*"

# Platforms that choose the port set PORT; 7860 is the default otherwise.
EXPOSE 7860
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers"]
