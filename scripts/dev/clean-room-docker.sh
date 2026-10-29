#!/bin/sh
# Open a shell in a clean Linux x86_64 container with a fresh clone of this repo inside (issue #337). For the CLI clients
# (Codex CLI, Gemini CLI, the Claude Code CLI). The image has git, curl and Node.js LTS, and no uv and no extra Python.
#
#   sh scripts/dev/clean-room-docker.sh [options] [-- command ...]
#
#   --ref REF       the commit, branch or tag to put in the container (default: HEAD of this checkout). Only committed work goes in.
#   --source DIR    the repo to clone from (default: this repo). A fresh clone is made on this computer, then copied into the
#                   container. The container has no mount of your folders and no credential of yours.
#   --url URL       clone URL inside the container instead of the local clone. Use a public repo. The URL is on the command line of this
#                   computer (a token in it would show in the process list), and the container needs the network.
#   --env NAME      pass the variable NAME from your environment into the container (an API key, for example). Repeat the option
#                   for more. Only these variables go in, and only at run time: the image never holds a key.
#   --rebuild       build the image again (the image is built when it is missing).
#   --keep          do not remove the container when you leave (its name is printed).
#   --image NAME    the image name (default: topopt-clean-room).
# The proxy variables of this computer (HTTP_PROXY, HTTPS_PROXY, NO_PROXY, ALL_PROXY and the lower-case forms), when set, go to the
# build and to the container by name: a host with no direct route needs them for apt-get, npm and git. A name that is set in one
# case only goes in both cases (apt reads only the lower-case names). No value is printed.
# Without a command, you get a bash shell. The container is removed when you leave (unless --keep).
# It needs the docker command and a running Docker daemon. This script changes nothing in your home.
# Nothing here uses bash features. It runs with the macOS /bin/sh and with dash.

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd -P) || exit 1
err() { printf 'clean-room-docker: %s\n' "$*" >&2; }
usage() { err "usage: clean-room-docker.sh [--ref REF] [--source DIR] [--url URL] [--env NAME]... [--rebuild] [--keep] [--image NAME] [-- command ...]"; exit 2; }

ref=HEAD; source=$ROOT; url=; envs=; rebuild=0; keep=0; image=topopt-clean-room
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) [ $# -ge 2 ] || usage; ref=$2; shift 2 ;;
    --source) [ $# -ge 2 ] || usage; source=$2; shift 2 ;;
    --url) [ $# -ge 2 ] || usage; url=$2; shift 2 ;;
    --env) [ $# -ge 2 ] || usage; envs="$envs $2"; shift 2 ;;
    --rebuild) rebuild=1; shift ;;
    --keep) keep=1; shift ;;
    --image) [ $# -ge 2 ] || usage; image=$2; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done

command -v docker >/dev/null 2>&1 || { err "the docker command is missing. Install Docker, or use 'clean-room.sh shell' instead."; exit 1; }
docker info >/dev/null 2>&1 || { err "no Docker daemon answers (docker info failed). Start Docker, then run this again."; exit 1; }
for v in $envs; do
  case "$v" in ''|[0-9]*|*[!A-Za-z0-9_]*) err "--env $v: not a variable name"; exit 1 ;; esac
  eval "isset=\${$v+x}"
  [ -n "$isset" ] || { err "--env $v: the variable is not set in your environment"; exit 1; }
done

# The proxy: only the names go on the command lines (docker reads each value from this environment), so no value is in the
# process list or in a log. Docker keeps the proxy build arguments out of the image history. apt reads only the lower-case names,
# and some tools read only the upper-case ones, so a name that is set in one case only is set in the other case too (same value).
for pair in HTTP_PROXY:http_proxy HTTPS_PROXY:https_proxy NO_PROXY:no_proxy ALL_PROXY:all_proxy; do
  up=${pair%%:*}; low=${pair#*:}
  eval "upval=\${$up-}; lowval=\${$low-}"
  if [ -n "$upval" ] && [ -z "$lowval" ]; then eval "$low=\$upval; export $low"
  elif [ -z "$upval" ] && [ -n "$lowval" ]; then eval "$up=\$lowval; export $up"
  fi
done
upval=; lowval=
proxies=; local_proxies=
for v in HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY http_proxy https_proxy no_proxy all_proxy; do
  eval "val=\${$v-}"
  [ -n "$val" ] || continue
  proxies="$proxies $v"
  case "$v" in NO_PROXY|no_proxy) continue ;; esac
  # The host of the proxy: no scheme, no path, no user:password@, no port.
  host=${val#*://}; host=${host%%/*}; host=${host##*@}
  case "$host" in \[*) host=${host%%]*}] ;; *) host=${host%%:*} ;; esac
  case "$host" in localhost|127.*|0.0.0.0|\[::1\]) local_proxies="$local_proxies $v" ;; esac
done
val=; host=
if [ -n "$proxies" ]; then
  err "passing the proxy variables$proxies of this computer to the build and to the container (the values are not shown)"
  [ -z "$local_proxies" ] || err "note:$local_proxies names an address on this computer (localhost). In the container that address is the container itself. Docker Desktop gives host.docker.internal for this computer. With Docker Engine on Linux this helper has no way to reach it: use a proxy address that a container can reach."
fi
buildargs=
for v in $proxies; do buildargs="$buildargs --build-arg $v"; done

if [ "$rebuild" -eq 1 ] || ! docker image inspect "$image" >/dev/null 2>&1; then
  err "building the image $image (the build context is docker/clean-room only)"
  # shellcheck disable=SC2086
  docker build --platform linux/amd64 $buildargs -t "$image" -f "$ROOT/docker/clean-room/Dockerfile" "$ROOT/docker/clean-room" || exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/clean-room-docker.XXXXXX") || exit 1
cid=
cleanup() {
  if [ -n "$cid" ] && [ "$keep" -eq 0 ]; then docker rm -f "$cid" >/dev/null 2>&1; fi
  rm -rf "$work/topopt"
  rmdir "$work" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ -z "$url" ]; then
  # A fresh clone on this computer (committed work only), then copied in. The origin is the real remote, not the local path.
  git clone -q "$source" "$work/topopt" || exit 1
  git -C "$work/topopt" checkout -q "$ref" || exit 1
  remote=$(git -C "$source" remote get-url origin 2>/dev/null) && git -C "$work/topopt" remote set-url origin "$remote"
fi

envargs=
for v in $proxies $envs; do envargs="$envargs -e $v"; done
# shellcheck disable=SC2086
cid=$(docker run -d --platform linux/amd64 $envargs "$image") || exit 1
if [ -n "$url" ]; then
  docker exec "$cid" git clone -q "$url" /home/tester/topopt || exit 1
  [ "$ref" = HEAD ] || docker exec -w /home/tester/topopt "$cid" git checkout -q "$ref" || exit 1
else
  tar -C "$work" -cf - topopt | docker exec -i "$cid" tar -xf - -C /home/tester || exit 1
fi
printf 'clean-room container: %s\nThe repo is in /home/tester/topopt. Leave with exit.%s\n' "$cid" \
  "$([ "$keep" -eq 1 ] && printf ' The container stays (docker rm -f %s).' "$cid")"
if [ $# -gt 0 ]; then
  if [ -t 0 ]; then tty=-it; else tty=; fi
  # shellcheck disable=SC2086
  docker exec $tty -w /home/tester/topopt "$cid" "$@"
else
  if [ -t 0 ]; then tty=-it; else tty=-i; fi
  docker exec $tty -w /home/tester/topopt "$cid" bash -l
fi
