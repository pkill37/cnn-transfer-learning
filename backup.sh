#! /bin/bash
set -euo pipefail

directory=$(echo "$1")
tarball=$(echo "$(basename $directory).tar.gz")

# Compress directory
tar -czvf "$tarball" "$directory"

# Print command to transfer tarball from local machine
echo "scp deeplar:$(readlink -f $tarball) ."
