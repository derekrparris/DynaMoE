#!/bin/sh
#
# Re-sign a locally built DynaMoE.app so it uses the data protection keychain.
#
#   Tools/dev-sign-dp.sh <path-to-DynaMoE.app> [--profile <file.provisionprofile>]
#
# Why: Debug builds deliberately carry no keychain-access-groups entitlement, so
# they stay on the legacy login keychain and keep asking for access on every new
# build (that is the ACL matching a code signature, not an app bug). This script
# puts the Release entitlement and an embedded Developer ID profile onto an
# app that is already built, which is the only way to exercise the shipped
# keychain path from a local build.
#
# The app is modified in place. Copy it somewhere disposable first if you care
# about the original signature.
#
# Read-only with respect to source: nothing here touches the project or Xcode.
#
# Note: this signs with the Release entitlements, which do not include
# get-task-allow, so the result is not debuggable. Running it is the point.

set -eu

if [ $# -lt 1 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

APP="$1"
shift
PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENTITLEMENTS="$ROOT/DynaMoE/DynaMoE/DynaMoE-Release.entitlements"
KEYCHAIN_GROUP="RMY84Y8G7M.DRP.DynaMoE"

[ -d "$APP" ] || { echo "error: app not found at $APP" >&2; exit 1; }
[ -f "$ENTITLEMENTS" ] || { echo "error: missing $ENTITLEMENTS" >&2; exit 1; }

IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | grep 'Developer ID Application' | head -1 | sed 's/^ *[0-9]*) [0-9A-F]* //')"
[ -n "$IDENTITY" ] || { echo "error: no 'Developer ID Application' identity in the keychain" >&2; exit 1; }

if [ -z "$PROFILE" ]; then
    for DIR in "$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles" \
               "$HOME/Library/MobileDevice/Provisioning Profiles"; do
        [ -d "$DIR" ] || continue
        for CANDIDATE in "$DIR"/*.provisionprofile; do
            [ -f "$CANDIDATE" ] || continue
            TEXT="$(security cms -D -i "$CANDIDATE" 2>/dev/null || true)"
            case "$TEXT" in
                *"$KEYCHAIN_GROUP"*) PROFILE="$CANDIDATE"; break 2 ;;
            esac
        done
    done
fi
[ -n "$PROFILE" ] || {
    echo "error: no installed provisioning profile grants $KEYCHAIN_GROUP." >&2
    echo "       Create a Developer ID profile for App ID DRP.DynaMoE, install it," >&2
    echo "       or pass it with --profile <file.provisionprofile>." >&2
    exit 1
}

echo "==> identity: $IDENTITY"
echo "==> profile:  $PROFILE"
cp "$PROFILE" "$APP/Contents/embedded.provisionprofile"

# codesign expands no build-setting variables, so the group is written literally
# here instead of shipping $(AppIdentifierPrefix) to the signer.
EXPANDED="$(mktemp -t dynamoe-dp-entitlements)"
trap 'rm -f "$EXPANDED"' EXIT
sed 's|\$(AppIdentifierPrefix)|RMY84Y8G7M|g' "$ENTITLEMENTS" > "$EXPANDED"

echo "==> signing (nested code first, then the app)"
find "$APP/Contents/Frameworks" -name '*.dylib' -o -name '*.framework' 2>/dev/null | while read -r NESTED; do
    codesign --force --timestamp --options runtime --sign "$IDENTITY" "$NESTED"
done
codesign --force --timestamp --options runtime \
    --entitlements "$EXPANDED" \
    --sign "$IDENTITY" "$APP"

echo "==> verifying"
codesign --verify --strict --verbose=2 "$APP"
codesign -d --entitlements :- "$APP" 2>&1 | grep -q "$KEYCHAIN_GROUP" \
    || { echo "error: $KEYCHAIN_GROUP missing from the signed app" >&2; exit 1; }
echo "ready: launch $APP and check Console.app for keychain backend=dataProtection"
