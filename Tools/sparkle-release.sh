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

# Canonicalize relative input paths NOW, before the script cd's elsewhere
# (packing/DMG steps run from other directories later on).
abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/$1" ;;
    esac
}

# keychain-access-groups is a restricted entitlement: the signed app needs an
# embedded Developer ID profile granting it, or launchd kills it at startup.
# The two halves are checked together here because either one alone ships a
# broken update -- the entitlement without a profile is a dead app, and the
# profile without the entitlement is the per-update keychain prompt.
KEYCHAIN_GROUP="RMY84Y8G7M.DRP.DynaMoE"
KEYCHAIN_TEAM="${KEYCHAIN_GROUP%%.*}"

# True when the profile's own keychain-access-groups list covers the group the
# app claims. Searching the profile's text would match for the wrong reason:
# com.apple.application-identifier is "<team>.<bundle>" too, so a profile whose
# access groups do not cover this app still contains the string.
profile_grants_group() {
    _plist="$(mktemp -t dynamoe-profile)"
    _groups="$(mktemp -t dynamoe-profile-groups)"
    if ! security cms -D -i "$1" >"$_plist" 2>/dev/null; then
        rm -f "$_plist" "$_groups"
        return 1
    fi
    plutil -extract 'Entitlements.keychain-access-groups' xml1 \
        -o "$_groups" "$_plist" 2>/dev/null || : >"$_groups"
    if grep -Fq "<string>$KEYCHAIN_GROUP</string>" "$_groups" \
        || grep -Fq "<string>$KEYCHAIN_TEAM.*</string>" "$_groups"; then
        rm -f "$_plist" "$_groups"
        return 0
    fi
    rm -f "$_plist" "$_groups"
    return 1
}

# The profile's ExpirationDate as an epoch second, or empty when the profile
# cannot be read or carries no date.
profile_expiration_epoch() {
    _plist="$(mktemp -t dynamoe-profile-expiry)"
    _epoch=""
    if security cms -D -i "$1" >"$_plist" 2>/dev/null; then
        _raw="$(plutil -extract ExpirationDate raw -o - "$_plist" 2>/dev/null || true)"
        if [ -n "$_raw" ]; then
            _epoch="$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$_raw" "+%s" 2>/dev/null || true)"
        fi
    fi
    rm -f "$_plist"
    printf '%s\n' "$_epoch"
}

# Refuses a profile that would ship a broken update: unreadable, not covering
# this app's keychain group, or expired (an expired profile validates, installs
# and notarizes, and then launchd kills the app on first launch).
check_profile() {
    if ! security cms -D -i "$1" >/dev/null 2>&1; then
        echo "error: $1 is not a readable provisioning profile." >&2
        exit 1
    fi
    if ! profile_grants_group "$1"; then
        echo "error: $1 does not grant keychain-access-groups = $KEYCHAIN_GROUP (or $KEYCHAIN_TEAM.*)." >&2
        echo "       The app would launch, then fail every keychain write." >&2
        exit 1
    fi
    _expiry_epoch="$(profile_expiration_epoch "$1")"
    if [ -z "$_expiry_epoch" ]; then
        echo "       note: $1 carries no ExpirationDate; cannot tell how long it lasts." >&2
        return 0
    fi
    # DYNAMOE_PROFILE_NOW_EPOCH exists so the expired and expiring-soon branches
    # can be exercised without waiting for a real profile to lapse.
    _now_epoch="${DYNAMOE_PROFILE_NOW_EPOCH:-$(date -u '+%s')}"
    _expiry_shown="$(date -u -r "$_expiry_epoch" '+%Y-%m-%d' 2>/dev/null || printf '%s' "$_expiry_epoch")"
    if [ "$_expiry_epoch" -le "$_now_epoch" ]; then
        echo "error: $1 expired on $_expiry_shown." >&2
        echo "       An expired profile still signs, notarizes and installs; the app is" >&2
        echo "       then killed at launch. Regenerate it against the installed certificate." >&2
        exit 1
    fi
    if [ "$_expiry_epoch" -le "$((_now_epoch + 2592000))" ]; then
        echo "warning: $1 expires on $_expiry_shown; renew it before it lapses." >&2
    fi
}

# Both halves of one app's keychain setup: the signed entitlement and the
# embedded profile that has to grant it. Applies to an exported app and to one
# this script just built, so it lives in one place.
check_app_keychain_path() {
    _app="$1"
    _entitled="$(codesign -d --entitlements :- "$_app" 2>&1 || true)"
    case "$_entitled" in
        *"$KEYCHAIN_GROUP"*) ;;
        *keychain-access-groups*)
            echo "error: '$_app' carries a keychain-access-groups that is not $KEYCHAIN_GROUP." >&2
            echo "       codesign expands no build variables, so a hand-signed build must use the literal group." >&2
            exit 1
            ;;
        *)
            echo "error: '$_app' has no keychain-access-groups entitlement." >&2
            echo "       Users would be asked to authorize keychain access again on every update." >&2
            echo "       Sign it with the Release entitlements and the 'DynaMoE Developer ID' profile." >&2
            exit 1
            ;;
    esac
    if [ ! -f "$_app/Contents/embedded.provisionprofile" ]; then
        echo "error: '$_app' claims keychain-access-groups but carries no embedded.provisionprofile." >&2
        echo "       That pairing refuses to launch; sign with the 'DynaMoE Developer ID' profile." >&2
        exit 1
    fi
    check_profile "$_app/Contents/embedded.provisionprofile"
}

# Validate all inputs up front so a bad path fails before any build/pack work
if [ -n "$APP_PATH" ]; then
    APP_PATH="$(abspath "$APP_PATH")"
    if [ ! -d "$APP_PATH" ]; then
        echo "error: app not found at $APP_PATH" >&2
        exit 1
    fi
fi
if [ -n "$ARCHIVE_PATH" ]; then
    ARCHIVE_PATH="$(abspath "$ARCHIVE_PATH")"
    if [ ! -f "$ARCHIVE_PATH" ]; then
        echo "error: archive not found at $ARCHIVE_PATH" >&2
        exit 1
    fi
fi
if [ -n "$DMG_PATH" ]; then
    DMG_PATH="$(abspath "$DMG_PATH")"
    if [ ! -f "$DMG_PATH" ]; then
        echo "error: DMG not found at $DMG_PATH" >&2
        exit 1
    fi
fi

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
    if ! codesign -dv --verbose=2 "$APP_PATH" 2>&1 | grep -q "Authority=Developer ID Application"; then
        echo "warning: '$APP_PATH' does not appear to be signed with Developer ID." >&2
        echo "         Updates installed from it may be blocked by Gatekeeper." >&2
    fi
    # keychain-access-groups is a restricted entitlement: the app needs an
    # embedded Developer ID profile that grants it, or launchd kills it at
    # startup. Shipping the entitlement without the profile ships a dead app,
    # and shipping the app without the entitlement is the per-update keychain
    # prompt this release exists to remove -- so neither half is a warning.
    check_app_keychain_path "$APP_PATH"
    if ! spctl --assess --type execute "$APP_PATH" >/dev/null 2>&1; then
        echo "warning: '$APP_PATH' does not pass Gatekeeper (notarize + staple it first)." >&2
    fi
else
    echo "==> Building Release app (dev build; use --app for distribution)..."
    # Release is manually signed against PROVISIONING_PROFILE_SPECIFIER, so
    # -allowProvisioningUpdates cannot resolve a profile by name; the profile has
    # to be installed beforehand (Tools/install_profile.py). Say so here instead
    # of failing later with "No profile for team ... found".
    INSTALLED_PROFILE=""
    for PROFILE_DIR in "$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles" \
                       "$HOME/Library/MobileDevice/Provisioning Profiles"; do
        [ -d "$PROFILE_DIR" ] || continue
        for CANDIDATE in "$PROFILE_DIR"/*.provisionprofile; do
            [ -f "$CANDIDATE" ] || continue
            if profile_grants_group "$CANDIDATE"; then INSTALLED_PROFILE="$CANDIDATE"; break 2; fi
        done
    done
    if [ -z "$INSTALLED_PROFILE" ]; then
        echo "error: no installed provisioning profile grants $KEYCHAIN_GROUP." >&2
        echo "       Release needs a Developer ID profile for App ID DRP.DynaMoE." >&2
        echo "       Install it with Tools/install_profile.py <file.provisionprofile>." >&2
        exit 1
    fi
    echo "==> profile:  $INSTALLED_PROFILE"
    check_profile "$INSTALLED_PROFILE"
    xcodebuild -project "$APP_DIR/DynaMoE.xcodeproj" \
        -scheme DynaMoE -configuration Release \
        -derivedDataPath "$OUT_DIR/DerivedData" \
        ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
        build
    APP_PATH="$OUT_DIR/DerivedData/Build/Products/Release/DynaMoE.app"
    # The build either signed with the Release entitlements and resolved the
    # profile, or it failed. Check anyway: this is the artifact that ships.
    if [ -d "$APP_PATH" ]; then
        check_app_keychain_path "$APP_PATH"
    fi
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
if [ -n "$DMG_PATH" ]; then
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
            <sparkle:minimumSystemVersion>26.0</sparkle:minimumSystemVersion>
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
