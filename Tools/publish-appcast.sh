#!/bin/sh
#
# Publishes Tools/appcast/appcast.xml to the gh-pages branch of the repo.
#
# The appcast URL baked into the app is:
#   https://derekrparris.github.io/DynaMoE/appcast.xml
#
# Usage: Tools/publish-appcast.sh
#
# Prereqs: the gh-pages branch must exist (create it once with:
#   git switch -c gh-pages && git rm -rf . && git commit -m init --allow-empty && git push -u origin gh-pages
# )

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
APPCAST_FILE="$SCRIPT_DIR/appcast/appcast.xml"
WORK="$(mktemp -d)"

if [ ! -f "$APPCAST_FILE" ]; then
    echo "error: appcast.xml not found. Run Tools/sparkle-release.sh first." >&2
    exit 1
fi

git fetch origin gh-pages
git worktree add "$WORK" origin/gh-pages
cp "$APPCAST_FILE" "$WORK/appcast.xml"

cd "$WORK"
# An untracked appcast.xml (first publish) is "changed": git diff ignores
# untracked files, so also require the file to be tracked before skipping.
if git diff --quiet -- appcast.xml && git ls-files --error-unmatch -- appcast.xml >/dev/null 2>&1; then
    echo "appcast.xml unchanged; nothing to publish."
else
    git add appcast.xml
    git commit -m "Update appcast for latest release"
    git push origin "HEAD:gh-pages"
    echo "appcast.xml published to gh-pages."
fi

git worktree remove "$WORK" --force
cd "$SCRIPT_DIR"
git worktree prune
