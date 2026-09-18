#!/bin/sh
#
# Regenerates Tools/appcast/appcast.xml from header.xml + items/*.xml + footer.xml.
# Run this after hand-editing any item (e.g. adding release notes), then publish
# with Tools/publish-appcast.sh. No app rebuild or re-signing needed.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ITEMS_DIR="$SCRIPT_DIR/appcast/items"
APPCAST_FILE="$SCRIPT_DIR/appcast/appcast.xml"

mkdir -p "$ITEMS_DIR"

{
    cat "$SCRIPT_DIR/appcast/header.xml"
    # Newest first (item files are written newest-last by sparkle-release.sh,
    # so mtime ordering puts the latest release first)
    ls -t "$ITEMS_DIR"/*.xml 2>/dev/null | while IFS= read -r f; do cat "$f"; done
    cat "$SCRIPT_DIR/appcast/footer.xml"
} > "$APPCAST_FILE"

ITEM_COUNT="$(ls "$ITEMS_DIR"/*.xml 2>/dev/null | wc -l | tr -d ' ')"
echo "wrote $APPCAST_FILE (${ITEM_COUNT:-0} items)"
