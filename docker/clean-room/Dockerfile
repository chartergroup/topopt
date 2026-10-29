# A clean room for the install tests of the CLI clients: Ubuntu 24.04 for Linux x86_64 with what a new user has
# before topopt. It has git, curl and Node.js LTS (for Codex CLI and Gemini CLI). It has NO uv and NO extra Python:
# topopt downloads its own pinned uv, and uv downloads Python 3.13.
# No key and no login is in the image. Keys come in at run time with `scripts/dev/clean-room-docker.sh --env NAME`.
# The build context is this folder only (the repo is not copied into the image).
#   docker build --platform linux/amd64 -t topopt-clean-room docker/clean-room
FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git less \
 && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# A normal user with a home, like a new user. CLI clients install into the home (npm install -g works without sudo).
# ENV (not .bashrc), so that `docker exec ... npm install -g ...` and `docker exec ... codex` work with no shell profile.
RUN useradd --create-home --shell /bin/bash tester \
 && mkdir -p /home/tester/.npm-global/bin \
 && printf 'PS1="(clean-room) \\u@\\h:\\w\\$ "\n' >> /home/tester/.bashrc \
 && chown -R tester:tester /home/tester
ENV NPM_CONFIG_PREFIX=/home/tester/.npm-global
ENV PATH=/home/tester/.npm-global/bin:$PATH
USER tester
WORKDIR /home/tester
CMD ["sleep", "infinity"]
