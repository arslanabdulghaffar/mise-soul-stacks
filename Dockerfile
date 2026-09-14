# Compiled console and headless local MuJoCo worker.
FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=osmesa \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
COPY scripts ./scripts
COPY bench ./bench
COPY configs ./configs
COPY eval ./eval
COPY tests ./tests
COPY Makefile ./
# Runtime assets and evidence hashes use the source tree relative to __file__.
# Keep imports anchored to /app/src instead of copying the package to site-packages.
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -c constraints.txt -e '.[test]'

# The SO-101 model is required at build time: clone with --recurse-submodules.
COPY vendor/SO-ARM100 ./vendor/SO-ARM100
RUN python scripts/build_scene.py
COPY --from=frontend /web/dist ./web/dist

EXPOSE 8000

CMD ["python", "-m", "mise.cli", "serve", "--host", "0.0.0.0"]
