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
    # Newest first: sort by each item's <sparkle:version> (CFBundleVersion,
    # dot-separated numbers) instead of file mtime, which is unstable across
    # fresh checkouts. Versions are zero-padded per component so a fixed-width
    # lexicographic sort equals numeric sort ("0.8.10" > "0.8.9", "26" < "26.1").
    ls "$ITEMS_DIR"/*.xml 2>/dev/null | while IFS= read -r f; do
        V="$(sed -n 's/.*<sparkle:version>\(.*\)<\/sparkle:version>.*/\1/p' "$f")"
        KEY="$(printf '%s' "$V" | awk -F. '{ printf "%06d.%06d.%06d.%06d", $1, $2, $3, $4 }')"
        printf '%s\t%s\n' "$KEY" "$f"
    done | sort -r | cut -f2- | while IFS= read -r f; do
        cat "$f"
    done
    cat "$SCRIPT_DIR/appcast/footer.xml"
} > "$APPCAST_FILE"

ITEM_COUNT="$(ls "$ITEMS_DIR"/*.xml 2>/dev/null | wc -l | tr -d ' ')"
echo "wrote $APPCAST_FILE (${ITEM_COUNT:-0} items)"
