#!/usr/bin/env sh

set -e

HOOK_NAME=$(basename "$0")
STAGED_FILES=$(git diff --staged --name-only)
STAGED_EXISTING_FILES=$(git diff --staged --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ] && [ "$HOOK_NAME" = "pre-commit" ]; then
  exit 0
fi

COMMIT_MSG_FILE="$1"
HOOK_ARGS=${COMMIT_MSG_FILE:-$STAGED_EXISTING_FILES}

TS_DIR="typescript"
HAS_TS_FILES=$(echo "$STAGED_FILES" | grep -q "^$TS_DIR/" && echo 1 || echo 0)

PY_DIR="python"
HAS_PY_FILES=$(echo "$STAGED_FILES" | grep -q "^$PY_DIR/" && echo 1 || echo 0)

# Run hooks based on staged files
if [ "$HAS_TS_FILES" -eq 1  ] && grep -q "\"git:$HOOK_NAME\"" "$TS_DIR/package.json"; then
  echo "Running $HOOK_NAME hook in $TS_DIR..."
  HOOK_ARGS_SCOPED=$(echo "$HOOK_ARGS" | sed 's/[^ ]* */..\/&/g' | sed 's/..\/typescript\///g')
  (cd "$TS_DIR" && npm run "git:$HOOK_NAME" "$HOOK_ARGS_SCOPED") || exit $?
  echo "$STAGED_EXISTING_FILES" | xargs -r git add
fi

if [ "$HAS_PY_FILES" -eq 1 ]; then
  echo "Running Python $HOOK_NAME hook..."
  HOOK_ARGS_SCOPED=$(echo "$HOOK_ARGS" | sed 's/[^ ]* */..\/&/g' | sed 's/..\/python\///g')
  (cd "$PY_DIR" && poetry run poe git --hook "$HOOK_NAME" "$HOOK_ARGS_SCOPED") || exit $?
  echo "$STAGED_EXISTING_FILES" | xargs -r git add
fi

# General commit style check
if [ "$HAS_TS_FILES" -eq 0 ] && [ "$HAS_PY_FILES" -eq 0 ]; then
  if [ "$HOOK_NAME" = "commit-msg" ] ; then
    COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")
    CONVENTIONAL_COMMIT_REGEX="^(feat|fix|chore|ci|docs|revert)(\(\w+\))?!?: .+"

    if ! echo "$COMMIT_MSG" | grep -qE "$CONVENTIONAL_COMMIT_REGEX"; then
      echo "Error: The provided commit message does not adhere to conventional commit style!"
      echo "Validation Regex: $CONVENTIONAL_COMMIT_REGEX"
      echo "Your commit message: $COMMIT_MSG"
      exit 1
    fi
  fi
fi
