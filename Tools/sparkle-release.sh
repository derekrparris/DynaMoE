#!/bin/sh
#
# Sparkle release helper for DynaMoE.
#
#   Tools/sparkle-release.sh <version> [--dmg <path-to-stapled-dmg>]
#
# What it does:
#   1. Builds the Release app via xcodebuild
#   2. Packs DynaMoE-<version>.app.tar.xz (the Sparkle update enclosure)
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
APPCAST_BASE="https://derekrparris.github.io/DynaMoE"
RELEASES_BASE="https://github.com/${REPO_SLUG}/releases/download"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
APP_DIR="$ROOT/DynaMoE"
OUT_DIR="${TMPDIR:-/tmp}/dynamoe-sparkle/$$"
TOOLS_DIR="$SCRIPT_DIR/sparkle-bin"
ITEMS_DIR="$SCRIPT_DIR/appcast/items"
APPCAST_FILE="$SCRIPT_DIR/appcast/appcast.xml"

if [ $# -eq 0 ]; then
    echo "usage: $0 <version> [--dmg <path>]" >&2
    exit 1
fi

VERSION="$1"
shift

DMG_PATH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dmg) DMG_PATH="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR" "$ITEMS_DIR"

# Fetch the Sparkle CLI tools once (sign_update)
if [ ! -x "$TOOLS_DIR/sign_update" ]; then
    echo "==> Fetching Sparkle CLI tools..."
    mkdir -p "$TOOLS_DIR"
    curl -sL -o "$TOOLS_DIR/sparkle.tar.xz" \
        "https://github.com/sparkle-project/Sparkle/releases/latest/download/Sparkle.tar.xz"
    tar -xJf "$TOOLS_DIR/sparkle.tar.xz" -C "$TOOLS_DIR" --strip-components=1 bin
    rm -f "$TOOLS_DIR/sparkle.tar.xz"
fi

# 1. Build the app
echo "==> Building Release app..."
xcodebuild -project "$APP_DIR/DynaMoE.xcodeproj" \
    -scheme DynaMoE -configuration Release \
    -derivedDataPath "$OUT_DIR/DerivedData" \
    ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
    build

APP_PATH="$OUT_DIR/DerivedData/Build/Products/Release/DynaMoE.app"
if [ ! -d "$APP_PATH" ]; then
    echo "error: built app not found at $APP_PATH" >&2
    exit 1
fi

SHORT_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_PATH/Contents/Info.plist")"
BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP_PATH/Contents/Info.plist")"

if [ "$SHORT_VERSION" != "$VERSION" ]; then
    echo "warning: requested version '$VERSION' but built app is '$SHORT_VERSION'" >&2
fi

PREV_VERSION_FILE="$ITEMS_DIR/.last_bundle_version"
if [ -f "$PREV_VERSION_FILE" ] && [ "$(cat "$PREV_VERSION_FILE")" = "$BUNDLE_VERSION" ]; then
    echo "error: CFBundleVersion ($BUNDLE_VERSION) was already served to users." >&2
    echo "       Bump CURRENT_PROJECT_VERSION in the Xcode project before releasing." >&2
    exit 1
fi

# 2. Pack the app archive
echo "==> Packing app archive..."
ARCHIVE_NAME="DynaMoE-$VERSION.app.tar.xz"
cd "$OUT_DIR/DerivedData/Build/Products/Release"
ditto -c -k --sequesterRsrc --keepParent DynaMoE.app "$OUT_DIR/$ARCHIVE_NAME"
cd "$OUT_DIR"

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
            <link>${APPCAST_BASE}/releases/v${VERSION}.html</link>
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

echo "$BUNDLE_VERSION" > "$PREV_VERSION_FILE"

# Assemble the feed: newest item first (items are sorted by pubDate descending)
{
    cat "$SCRIPT_DIR/appcast/header.xml"
    ls -t "$ITEMS_DIR"/*.xml 2>/dev/null | while IFS= read -r f; do cat "$f"; done
    cat "$SCRIPT_DIR/appcast/footer.xml"
} > "$APPCAST_FILE"

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
