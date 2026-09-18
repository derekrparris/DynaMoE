#!/bin/sh
#
# Sparkle release helper for DynaMoE.
#
#   Tools/sparkle-release.sh <version> [--app <path-to-notarized.app>]
#                                        [--archive <path-to-premade-archive>]
#                                        [--dmg <path-to-stapled-dmg>]
#
# What it does:
#   1. Takes your Developer ID-signed, notarized, stapled .app (from Xcode
#      Archive/Export) — or falls back to building via xcodebuild (dev-only)
#   2. Packs DynaMoE-<version>.app.tar.xz (the Sparkle update enclosure),
#      unless you supply a pre-made archive via --archive
#   3. EdDSA-signs the archive (private key lives in your login Keychain)
#   4. Adds an <item> to Tools/appcast/items/ and regenerates appcast.xml
#   5. Prints the `gh release create` command to publish everything
#
# Notes:
#   - appcast.xml enclosures point at GitHub release asset URLs, so publish
#     the release BEFORE (or at the same time as) shipping the appcast.
#   - CFBundleVersion (CURRENT_PROJECT_VERSION) must be bumped per release;
#     the script refuses to reuse a version Sparkle has already served.

set -eu

REPO_SLUG="derekrparris/DynaMoE"
RELEASES_BASE="https://github.com/${REPO_SLUG}/releases/download"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(dirname "$SCRIPT_DIR")"
APP_DIR="$ROOT/DynaMoE"
OUT_DIR="${TMPDIR:-/tmp}/dynamoe-sparkle/$$"
TOOLS_DIR="$SCRIPT_DIR/sparkle-bin"
ITEMS_DIR="$SCRIPT_DIR/appcast/items"
APPCAST_FILE="$SCRIPT_DIR/appcast/appcast.xml"

if [ $# -eq 0 ]; then
    echo "usage: $0 <version> [--app <path>] [--archive <path>] [--dmg <path>]" >&2
    exit 1
fi

VERSION="$1"
shift

APP_PATH=""
ARCHIVE_PATH=""
DMG_PATH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --app) APP_PATH="$2"; shift 2 ;;
        --archive) ARCHIVE_PATH="$2"; shift 2 ;;
        --dmg) DMG_PATH="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR" "$ITEMS_DIR"

# Fetch the Sparkle CLI tools once (sign_update, generate_appcast, generate_keys)
SPARKLE_TOOLS_VERSION="2.9.6"
if [ ! -x "$TOOLS_DIR/sign_update" ]; then
    echo "==> Fetching Sparkle CLI tools..."
    mkdir -p "$TOOLS_DIR/.extract"
    curl -sfL -o "$TOOLS_DIR/sparkle.tar.xz" \
        "https://github.com/sparkle-project/Sparkle/releases/download/${SPARKLE_TOOLS_VERSION}/Sparkle-${SPARKLE_TOOLS_VERSION}.tar.xz" \
        || { rm -f "$TOOLS_DIR/sparkle.tar.xz"; echo "error: failed to download Sparkle tools" >&2; exit 1; }
    tar -xJf "$TOOLS_DIR/sparkle.tar.xz" -C "$TOOLS_DIR/.extract"
    cp "$TOOLS_DIR/.extract/bin/sign_update" "$TOOLS_DIR/.extract/bin/generate_appcast" "$TOOLS_DIR/.extract/bin/generate_keys" "$TOOLS_DIR/"
    rm -rf "$TOOLS_DIR/.extract" "$TOOLS_DIR/sparkle.tar.xz"
fi

# 1. Obtain the app: user-provided (notarized) or dev build
if [ -n "$APP_PATH" ]; then
    if [ ! -d "$APP_PATH" ]; then
        echo "error: app not found at $APP_PATH" >&2
        exit 1
    fi
    echo "==> Using provided app: $APP_PATH"
    if ! codesign -dv "$APP_PATH" 2>&1 | grep -q "Developer ID"; then
        echo "warning: '$APP_PATH' does not appear to be signed with Developer ID." >&2
        echo "         Updates installed from it may be blocked by Gatekeeper." >&2
    fi
    if ! spctl --assess --type execute "$APP_PATH" >/dev/null 2>&1; then
        echo "warning: '$APP_PATH' does not pass Gatekeeper (notarize + staple it first)." >&2
    fi
else
    echo "==> Building Release app (dev build; use --app for distribution)..."
    xcodebuild -project "$APP_DIR/DynaMoE.xcodeproj" \
        -scheme DynaMoE -configuration Release \
        -derivedDataPath "$OUT_DIR/DerivedData" \
        ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
        build
    APP_PATH="$OUT_DIR/DerivedData/Build/Products/Release/DynaMoE.app"
fi

if [ ! -d "$APP_PATH" ]; then
    echo "error: built app not found at $APP_PATH" >&2
    exit 1
fi

SHORT_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_PATH/Contents/Info.plist")"
BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP_PATH/Contents/Info.plist")"

if [ "$SHORT_VERSION" != "$VERSION" ]; then
    echo "error: requested version '$VERSION' but the app reports '$SHORT_VERSION'." >&2
    echo "       Rerun with the matching version, or rebuild with updated MARKETING_VERSION." >&2
    exit 1
fi

# Guard against re-serving a CFBundleVersion: check every recorded item
# (committed in Tools/appcast/items/), not just the most recent one, so the
# guard survives fresh checkouts and is authoritative once an item is committed.
for ITEM_FILE in "$ITEMS_DIR"/*.xml; do
    [ -f "$ITEM_FILE" ] || continue
    SERVED="$(sed -n 's/.*<sparkle:version>\(.*\)<\/sparkle:version>.*/\1/p' "$ITEM_FILE")"
    if [ "$SERVED" = "$BUNDLE_VERSION" ]; then
        echo "error: CFBundleVersion ($BUNDLE_VERSION) already appears in $(basename "$ITEM_FILE")." >&2
        echo "       Bump CURRENT_PROJECT_VERSION in the Xcode project before releasing." >&2
        exit 1
    fi
done

# 2. Obtain the update archive (pre-made via --archive, or pack the app)
if [ -n "$ARCHIVE_PATH" ]; then
    if [ ! -f "$ARCHIVE_PATH" ]; then
        echo "error: archive not found at $ARCHIVE_PATH" >&2
        exit 1
    fi
    echo "==> Using provided archive: $ARCHIVE_PATH"
    ARCHIVE_NAME="$(basename "$ARCHIVE_PATH")"
    case "$ARCHIVE_NAME" in
        *.tar.xz|*.tar.gz|*.tar.bz2|*.zip|*.dmg) ;;
        *) echo "warning: '$ARCHIVE_NAME' has an unrecognized extension." >&2
           echo "         Sparkle selects its extractor by file extension." >&2 ;;
    esac
    cp "$ARCHIVE_PATH" "$OUT_DIR/$ARCHIVE_NAME"
else
    echo "==> Packing app archive..."
    ARCHIVE_NAME="DynaMoE-$VERSION.app.tar.xz"
    APP_DIR_PATH="$(dirname "$APP_PATH")"
    cd "$APP_DIR_PATH"
    tar --no-xattrs -cJf "$OUT_DIR/$ARCHIVE_NAME" "$(basename "$APP_PATH")"
    cd "$OUT_DIR"
fi

# 3. Sign
echo "==> Signing with EdDSA..."
SIG_OUT="$("$TOOLS_DIR/sign_update" "$OUT_DIR/$ARCHIVE_NAME")"
echo "$SIG_OUT"
ARCHIVE_SIG="$(echo "$SIG_OUT" | sed -n 's/.*sparkle:edSignature="\([^"]*\)".*/\1/p')"
ARCHIVE_LEN="$(echo "$SIG_OUT" | sed -n 's/.*length="\([0-9]*\)".*/\1/p')"

if [ -z "$ARCHIVE_SIG" ] || [ -z "$ARCHIVE_LEN" ]; then
    echo "error: failed to parse sign_update output" >&2
    exit 1
fi

DMG_NAME=""
DMG_CMD=""
if [ -n "$DMG_PATH" ] && [ -f "$DMG_PATH" ]; then
    DMG_NAME="DynaMoE-$VERSION.dmg"
    cp "$DMG_PATH" "$OUT_DIR/$DMG_NAME"
    DMG_CMD=" '$OUT_DIR/$DMG_NAME'"
fi

# 4. Record the item and regenerate appcast.xml
PUB_DATE="$(date -u "+%a, %d %b %Y %H:%M:%S %z")"
cat > "$ITEMS_DIR/$VERSION.xml" <<EOF
        <item>
            <title>Version $VERSION</title>
            <sparkle:version>$BUNDLE_VERSION</sparkle:version>
            <sparkle:shortVersionString>$VERSION</sparkle:shortVersionString>
            <link>https://github.com/${REPO_SLUG}/releases/tag/v${VERSION}</link>
            <description><![CDATA[DynaMoE $VERSION]]></description>
            <pubDate>$PUB_DATE</pubDate>
            <enclosure
                url="$RELEASES_BASE/v$VERSION/$ARCHIVE_NAME"
                sparkle:edSignature="$ARCHIVE_SIG"
                length="$ARCHIVE_LEN"
                type="application/octet-stream"
            />
        </item>
EOF

# Assemble the feed
"$SCRIPT_DIR/generate-appcast.sh"

# 5. Report
echo ""
echo "==> Artifacts ready in $OUT_DIR:"
ls -la "$OUT_DIR"
echo ""
echo "==> Next steps:"
echo "  1. Publish the release (DMG you pass in should already be notarized + stapled):"
echo "     gh release create v$VERSION '$OUT_DIR/$ARCHIVE_NAME'${DMG_CMD} --title 'DynaMoE $VERSION' --notes '...'"
echo "  2. After the release is public, publish the appcast:"
echo "     Tools/publish-appcast.sh"
