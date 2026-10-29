#!/bin/bash
# Copy the user files of this case (files/ beside this script) into the empty workspace.
# The runner starts it as `bash <absolute path>` with the workspace as the current folder,
# and only when `claude plugin eval` gets --scaffold.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p files
cp "$here"/files/*.yaml "$here"/files/*.stl files/
