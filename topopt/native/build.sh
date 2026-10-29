#!/bin/sh
# Build the Accelerate Cholesky shim. Apple silicon only.
cd "$(dirname "$0")" || exit 1
xcrun clang -O2 -shared -fPIC -framework Accelerate -o libaccel_chol.dylib accel_chol.c
