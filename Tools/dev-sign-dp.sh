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

# find-identity prints the name inside double quotes, and those quotes are part
# of what sed/awk would capture: passing them to codesign makes it look for a
# name that starts with a quote and answer "no identity found". Take the text
# between the quotes instead.
IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | awk -F'"' '/Developer ID Application/ { print $2; exit }')"
[ -n "$IDENTITY" ] || { echo "error: no 'Developer ID Application' identity in the keychain" >&2; exit 1; }

# True when a profile's own keychain-access-groups list covers the group this
# app claims. Searching the profile's text would match for the wrong reason:
# com.apple.application-identifier is "TEAM.DRP.DynaMoE" too, so a profile whose
# access groups do not cover the app still contains the string.
KEYCHAIN_TEAM="${KEYCHAIN_GROUP%%.*}"
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

if [ -z "$PROFILE" ]; then
    for DIR in "$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles" \
               "$HOME/Library/MobileDevice/Provisioning Profiles"; do
        [ -d "$DIR" ] || continue
        for CANDIDATE in "$DIR"/*.provisionprofile; do
            [ -f "$CANDIDATE" ] || continue
            if profile_grants_group "$CANDIDATE"; then PROFILE="$CANDIDATE"; break 2; fi
        done
    done
else
    [ -f "$PROFILE" ] || { echo "error: no such profile: $PROFILE" >&2; exit 1; }
    profile_grants_group "$PROFILE" || {
        echo "error: $PROFILE does not grant $KEYCHAIN_GROUP (or $KEYCHAIN_TEAM.*)." >&2
        echo "       The app would be killed at launch for claiming a group its profile does not cover." >&2
        exit 1
    }
fi
[ -n "$PROFILE" ] || {
    echo "error: no installed provisioning profile grants $KEYCHAIN_GROUP." >&2
    echo "       Create a Developer ID profile for App ID DRP.DynaMoE, install it with" >&2
    echo "       Tools/install_profile.py, or pass it with --profile <file.provisionprofile>." >&2
    exit 1
}

echo "==> identity: $IDENTITY"
echo "==> profile:  $PROFILE"
cp "$PROFILE" "$APP/Contents/embedded.provisionprofile"

# codesign expands no build-setting variables, so the group is written literally
# here instead of shipping $(AppIdentifierPrefix) to the signer. Xcode's value
# for that variable includes its trailing dot ($(AppIdentifierPrefix)DRP.DynaMoE
# is "RMY84Y8G7M.DRP.DynaMoE"), so the replacement has to include it too.
EXPANDED="$(mktemp -t dynamoe-dp-entitlements)"
NESTED_LIST="$(mktemp -t dynamoe-nested-code)"
trap 'rm -f "$EXPANDED" "$NESTED_LIST"' EXIT
sed 's|\$(AppIdentifierPrefix)|RMY84Y8G7M.|g' "$ENTITLEMENTS" > "$EXPANDED"

echo "==> signing (nested code first, then the app)"
# Every piece of nested code is signed before the bundle that contains it, and
# the list of them covers XPC services and helper apps (Sparkle ships both)
# rather than only dylibs and frameworks: re-signing an app leaves a nested
# bundle's old signature in place, and under the hardened runtime that is a
# launch or load failure.
#
# The list goes through a file rather than a pipe on purpose. A pipeline runs
# the loop in a subshell, where a failed codesign could not abort the script.
# `! -type l` skips the versioned framework's top-level symlinks (Updater.app,
# Sparkle, ...), which resolve to a bundle this loop already signed.
find "$APP/Contents" -depth \
    \( -name '*.dylib' -o -name '*.so' -o -name '*.framework' \
       -o -name '*.xpc' -o -name '*.app' \) ! -type l 2>/dev/null > "$NESTED_LIST"
while IFS= read -r NESTED; do
    [ -n "$NESTED" ] || continue
    echo "    $NESTED"
    if ! codesign --force --timestamp --options runtime --sign "$IDENTITY" "$NESTED"; then
        echo "error: could not sign nested code at $NESTED" >&2
        exit 1
    fi
done < "$NESTED_LIST"
codesign --force --timestamp --options runtime \
    --entitlements "$EXPANDED" \
    --sign "$IDENTITY" "$APP"

echo "==> verifying"
codesign --verify --strict --verbose=2 "$APP"
codesign -d --entitlements :- "$APP" 2>&1 | grep -q "$KEYCHAIN_GROUP" \
    || { echo "error: $KEYCHAIN_GROUP missing from the signed app" >&2; exit 1; }
echo "ready: launch $APP and check Console.app for keychain backend=dataProtection"
