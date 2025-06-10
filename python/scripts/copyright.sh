#!/bin/bash
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

set -e

if [ "$#" -eq 0 ]; then
  TARGETS=(beeai_framework/**/*.py cz_commitizen/*.py tests/**/*.py scripts/*.{sh,py})
else
  TARGETS=("${@/#$PWD\//}")
fi

AUTHOR="© BeeAI a Series of LF Projects, LLC"

# Check if 'nwa' command is not available and 'brew' is available
if ! command -v nwa &> /dev/null && command -v brew &> /dev/null; then
  echo "Installing 'nwa' via 'brew' (https://github.com/B1NARY-GR0UP/nwa)"
  brew tap B1NARY-GR0UP/nwa
  brew install nwa
fi

# Check if 'nwa' command is not available and 'go' is available, then install 'nwa'
if ! command -v nwa &> /dev/null && command -v go &> /dev/null; then
  echo "Installing 'nwa' via 'go' (https://github.com/B1NARY-GR0UP/nwa)"
  go install github.com/B1NARY-GR0UP/nwa@latest
  # Ensure the GOPATH is added to the PATH environment variable
  export PATH=$PATH:$(go env GOPATH)/bin
fi

TYPE=${TYPE:-add}

if command -v nwa &> /dev/null; then
  echo "Running 'nwa' version $(nwa --version)"
  nwa "${TYPE}" -i "Apache-2.0" -c "$AUTHOR" "${TARGETS[@]}"
elif command -v docker &> /dev/null; then
  docker run --rm -v "${PWD}:/src" ghcr.io/b1nary-gr0up/nwa:main "${TYPE}" -i "Apache-2.0" -c "$AUTHOR" "${TARGETS[@]}"
else
  if [ "$COPYRIGHT_STRICT" = true ] ; then
    echo "Error: 'nwa' is not available. Either install it manually or install go/docker."
    exit 1
  else
    echo "Copyright script was not executed because the nwa package could not be installed."
  fi
fi
