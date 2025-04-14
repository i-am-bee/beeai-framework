#!/bin/bash
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [ "$#" -eq 0 ]; then
  TARGETS=('src/**/*.{js,ts,proto}' "dist/**/*.js" "tests/**/*.{js,ts}" "scripts/**/*.{sh,ts,js}")
else
  TARGETS=("${@/#$PWD\//}")
fi

# Path to the package.json file
PACKAGE_JSON_PATH="./package.json"

# Check if the package.json file exists
if [[ ! -f "$PACKAGE_JSON_PATH" ]]; then
  echo "Error: package.json file not found at $PACKAGE_JSON_PATH"
  exit 1
fi

# Retrieve the author property using jq
AUTHOR=$(jq -r '.author' "$PACKAGE_JSON_PATH")

# Check if the author property is not null or empty
if [[ -z "$AUTHOR" ]]; then
  echo "Error: author property not found in package.json"
  exit 1
fi

AUTHOR="© $AUTHOR"

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
  nwa "${TYPE}" -l apache -c "$AUTHOR" "${TARGETS[@]}"
elif command -v docker &> /dev/null; then
  docker run --rm -v "${PWD}:/src" ghcr.io/b1nary-gr0up/nwa:main "${TYPE}" -l apache -c "$AUTHOR" "${TARGETS[@]}"
else
  if [ "$COPYRIGHT_STRICT" = true ] ; then
    echo "Error: 'nwa' is not available. Either install it manually or install go/docker."
    exit 1
  else
    echo "Copyright script was not executed because the nwa package could not be installed."
  fi
fi
