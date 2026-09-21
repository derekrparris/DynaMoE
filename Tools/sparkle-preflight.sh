#!/bin/sh
#
# Pre-release checks for DynaMoE Sparkle releases.
#
#   Tools/sparkle-preflight.sh [--app <path-to-exported.app>]
#                              [--version <x.y.z>]
#                              [--profile <notarytool-profile>]
#                              [--offline]
#
# Read-only: verifies prerequisites and changes nothing. Exits non-zero if any
# gate fails, so a release fails here instead of part-way through notarizing.
#
# Gates:
#   1. Working tree has the app (catches being stranded on an orphan branch)
#   2. Info.plist carries the Sparkle keys
#   3. Sparkle CLI tools are present in Tools/sparkle-bin/
#   4. The Keychain private key matches SUPublicEDKey  <- the critical one
#   5. A Developer ID Application identity exists
#   6. The notarytool credential profile exists
#   7. The app target's version has not already been served
#   8. appcast.xml is well-formed
#   9. gh-pages exists (and, unless --offline, the feed URL is reachable)
#  10. With --app: signature, Gatekeeper and staple checks on the export
#
# Note: gate 4 can raise a Keychain authorization dialog on first use. Approve
# it — a dismissed dialog makes generate_keys succeed while printing nothing,
# which would look like a key mismatch.

set -eu

REPO_SLUG="derekrparris/DynaMoE"
PAGES_BASE="https://derekrparris.github.io/DynaMoE"
# Matches the profile recorded in _Builds/release.txt. Override with --profile
# or DYNAMOE_NOTARY_PROFILE.
NOTARY_PROFILE="${DYNAMOE_NOTARY_PROFILE:-dynamoe-profile}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ROOT="$(dirname "$SCRIPT_DIR")"
PLIST="$ROOT/DynaMoE/DynaMoE/Info.plist"
PBXPROJ="$ROOT/DynaMoE/DynaMoE.xcodeproj/project.pbxproj"
TOOLS_DIR="$SCRIPT_DIR/sparkle-bin"
ITEMS_DIR="$SCRIPT_DIR/appcast/items"
APPCAST_FILE="$SCRIPT_DIR/appcast/appcast.xml"

APP_PATH=""
WANT_VERSION=""
ONLINE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --app)     APP_PATH="${2:-}"; shift 2 ;;
        --version) WANT_VERSION="${2:-}"; shift 2 ;;
        --profile) NOTARY_PROFILE="${2:-}"; shift 2 ;;
        --offline) ONLINE=0; shift ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

FAILURES=0
WARNINGS=0

ok()   { printf '  ok    %s\n' "$1"; }
warn() { printf '  warn  %s\n' "$1"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '  FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
skip() { printf '  skip  %s\n' "$1"; }
head_() { printf '\n%s\n' "$1"; }

# ---------------------------------------------------------------- 1. tree
head_ "1. Working tree"
if [ -f "$PLIST" ] && [ -d "$ROOT/DynaMoE/DynaMoE" ]; then
    ok "app sources present"
else
    fail "app sources missing — are you on gh-pages? Recover with: git switch main"
fi

# ------------------------------------------------------------ 2. Info.plist
head_ "2. Info.plist Sparkle keys"
if [ ! -f "$PLIST" ]; then
    skip "no Info.plist to read"
else
    PLIST_PUBKEY="$(/usr/libexec/PlistBuddy -c 'Print :SUPublicEDKey' "$PLIST" 2>/dev/null || true)"
    PLIST_FEED="$(/usr/libexec/PlistBuddy -c 'Print :SUFeedURL' "$PLIST" 2>/dev/null || true)"
    PLIST_AUTO="$(/usr/libexec/PlistBuddy -c 'Print :SUEnableAutomaticChecks' "$PLIST" 2>/dev/null || true)"
    [ -n "$PLIST_PUBKEY" ] && ok "SUPublicEDKey present" || fail "SUPublicEDKey missing"
    [ -n "$PLIST_FEED" ] && ok "SUFeedURL = $PLIST_FEED" || fail "SUFeedURL missing"
    [ "$PLIST_AUTO" = "true" ] && ok "SUEnableAutomaticChecks = true" || warn "SUEnableAutomaticChecks is '${PLIST_AUTO:-unset}'"
fi

# ----------------------------------------------------------- 3. CLI tools
head_ "3. Sparkle CLI tools"
for T in sign_update generate_keys; do
    if [ -x "$TOOLS_DIR/$T" ]; then
        ok "$T"
    else
        fail "$T missing from Tools/sparkle-bin/ (sparkle-release.sh downloads these on first run)"
    fi
done

# -------------------------------------------------------- 4. signing key
head_ "4. Signing key (the update chain)"
if [ ! -x "$TOOLS_DIR/generate_keys" ]; then
    skip "generate_keys unavailable"
elif [ -z "${PLIST_PUBKEY:-}" ]; then
    skip "no SUPublicEDKey to compare against"
else
    KEYCHAIN_PUBKEY="$("$TOOLS_DIR/generate_keys" -p 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -z "$KEYCHAIN_PUBKEY" ]; then
        fail "no key returned — the Keychain prompt may have been dismissed (re-run and approve it)"
    elif [ "$KEYCHAIN_PUBKEY" = "$PLIST_PUBKEY" ]; then
        ok "Keychain key matches SUPublicEDKey"
    else
        fail "KEY MISMATCH — Keychain key '$KEYCHAIN_PUBKEY' != plist '$PLIST_PUBKEY'"
        fail "DO NOT SHIP: existing installs would be unable to verify updates"
    fi
fi

# ----------------------------------------------------------- 5. identity
head_ "5. Code signing identity"
if security find-identity -v -p codesigning 2>/dev/null | grep -q 'Developer ID Application'; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | grep 'Developer ID Application' | head -1 | sed 's/^ *[0-9]*) [0-9A-F]* //')"
    ok "$IDENTITY"
else
    fail "no 'Developer ID Application' identity in the keychain"
fi

# ------------------------------------------------------- 6. notary profile
head_ "6. Notary credentials"
if [ "$ONLINE" -eq 0 ]; then
    skip "notarytool history needs network (--offline)"
else
    NOTARY_OUT="$(xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" 2>&1 || true)"
    case "$NOTARY_OUT" in
        *"No Keychain password item found for profile"*)
            fail "profile '$NOTARY_PROFILE' not found — run: xcrun notarytool store-credentials \"$NOTARY_PROFILE\"" ;;
        *"Successfully received submission history"*)
            ok "profile '$NOTARY_PROFILE' works" ;;
        *)
            warn "could not verify profile '$NOTARY_PROFILE' (offline?): $(printf '%s' "$NOTARY_OUT" | head -1)" ;;
    esac
fi

# ------------------------------------------------------ 7. version guard
head_ "7. Version not already served"
if [ ! -f "$PBXPROJ" ]; then
    skip "no project file to read"
else
    # App target values come first in the file; the test target also sets them,
    # so report if the values are not uniform.
    MV_ALL="$(grep -o 'MARKETING_VERSION = [^;]*;' "$PBXPROJ" | sed 's/.*= //; s/;//' | sort -u)"
    CPV_ALL="$(grep -o 'CURRENT_PROJECT_VERSION = [^;]*;' "$PBXPROJ" | sed 's/.*= //; s/;//' | sort -u)"
    MARKETING_VERSION="$(printf '%s\n' "$MV_ALL" | head -1)"
    BUNDLE_VERSION="$(printf '%s\n' "$CPV_ALL" | head -1)"
    ok "MARKETING_VERSION=$MARKETING_VERSION  CURRENT_PROJECT_VERSION=$BUNDLE_VERSION"
    if [ "$(printf '%s\n' "$MV_ALL" | wc -l | tr -d ' ')" -gt 1 ]; then
        warn "multiple MARKETING_VERSION values in the project: $(printf '%s' "$MV_ALL" | tr '\n' ' ')"
    fi
    if [ "$(printf '%s\n' "$CPV_ALL" | wc -l | tr -d ' ')" -gt 1 ]; then
        warn "multiple CURRENT_PROJECT_VERSION values in the project: $(printf '%s' "$CPV_ALL" | tr '\n' ' ')"
    fi

    case "$BUNDLE_VERSION" in
        *[!0-9.]*) fail "CURRENT_PROJECT_VERSION '$BUNDLE_VERSION' must be dot-separated numbers only" ;;
    esac

    if [ -n "$WANT_VERSION" ] && [ "$WANT_VERSION" != "$MARKETING_VERSION" ]; then
        fail "requested version '$WANT_VERSION' != project MARKETING_VERSION '$MARKETING_VERSION'"
    fi

    SERVED_MATCH=""
    for ITEM_FILE in "$ITEMS_DIR"/*.xml; do
        [ -f "$ITEM_FILE" ] || continue
        SERVED="$(sed -n 's/.*<sparkle:version>\(.*\)<\/sparkle:version>.*/\1/p' "$ITEM_FILE")"
        if [ "$SERVED" = "$BUNDLE_VERSION" ]; then
            SERVED_MATCH="$(basename "$ITEM_FILE")"
        fi
    done
    if [ -n "$SERVED_MATCH" ]; then
        fail "CFBundleVersion $BUNDLE_VERSION already appears in $SERVED_MATCH — bump CURRENT_PROJECT_VERSION"
    else
        ok "CFBundleVersion $BUNDLE_VERSION has not been served"
    fi
fi

# ---------------------------------------------------------- 8. appcast xml
head_ "8. Appcast"
if [ ! -f "$APPCAST_FILE" ]; then
    warn "Tools/appcast/appcast.xml not found"
elif command -v xmllint >/dev/null 2>&1; then
    if xmllint --noout "$APPCAST_FILE" 2>/dev/null; then
        ITEM_COUNT="$(grep -c '<sparkle:version>' "$APPCAST_FILE" || true)"
        ok "well-formed, ${ITEM_COUNT:-0} item(s)"
    else
        fail "appcast.xml is not well-formed — run xmllint Tools/appcast/appcast.xml"
    fi
else
    warn "xmllint unavailable; skipped well-formedness check"
fi

# -------------------------------------------------------------- 9. gh-pages
head_ "9. Feed hosting"
if git -C "$ROOT" show-ref --verify --quiet refs/heads/gh-pages 2>/dev/null \
   || git -C "$ROOT" show-ref --verify --quiet refs/remotes/origin/gh-pages 2>/dev/null; then
    ok "gh-pages branch exists"
else
    fail "no gh-pages branch (local or origin) — the appcast has nowhere to live"
fi
if [ "$ONLINE" -eq 0 ]; then
    skip "URL reachability needs network (--offline)"
elif [ -n "${PLIST_FEED:-}" ] && command -v curl >/dev/null 2>&1; then
    CODE="$(curl -s -o /dev/null -w '%{http_code}' "$PLIST_FEED" || echo 000)"
    case "$CODE" in
        200) ok "feed reachable ($PLIST_FEED)" ;;
        404) warn "feed 404 — expected before the first publish-appcast.sh, and after that a problem" ;;
        *)   warn "feed returned HTTP $CODE" ;;
    esac
else
    skip "no feed URL available"
fi

# ------------------------------------------------------------- 10. the app
head_ "10. Exported app"
if [ -z "$APP_PATH" ]; then
    skip "no --app given (pass the export to check signature + Gatekeeper + staple)"
elif [ ! -d "$APP_PATH" ]; then
    fail "--app path not found: $APP_PATH"
else
    if codesign --verify --deep --strict --verbose=2 "$APP_PATH" 2>/dev/null; then
        ok "codesign --verify --deep --strict"
    else
        fail "signature invalid or a nested helper is adhoc-signed (re-export from Xcode)"
    fi
    if spctl --assess --type execute "$APP_PATH" 2>/dev/null; then
        ok "Gatekeeper accepts the app"
    else
        fail "Gatekeeper rejected the app (notarize + staple it)"
    fi
    if xcrun stapler validate "$APP_PATH" 2>/dev/null | grep -q 'validate action worked'; then
        ok "notarization ticket stapled to the app"
    else
        fail "no stapled ticket on the app — Sparkle installs this .app, so staple it"
    fi
    APP_MV="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_PATH/Contents/Info.plist" 2>/dev/null || true)"
    APP_CPV="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP_PATH/Contents/Info.plist" 2>/dev/null || true)"
    ok "app reports $APP_MV ($APP_CPV)"
    if [ -n "$APP_MV" ] && [ -n "${MARKETING_VERSION:-}" ] && [ "$APP_MV" != "$MARKETING_VERSION" ]; then
        fail "app CFBundleShortVersionString '$APP_MV' != project '$MARKETING_VERSION'"
    fi
    if [ -n "${PLIST_PUBKEY:-}" ]; then
        APP_PUBKEY="$(/usr/libexec/PlistBuddy -c 'Print :SUPublicEDKey' "$APP_PATH/Contents/Info.plist" 2>/dev/null || true)"
        if [ "$APP_PUBKEY" = "$PLIST_PUBKEY" ]; then
            ok "app carries the expected SUPublicEDKey"
        else
            if [ -z "$APP_PUBKEY" ]; then
                fail "app has no SUPublicEDKey — it predates Sparkle, or Sparkle is disabled in that build"
            else
                fail "app's SUPublicEDKey ($APP_PUBKEY) does not match the source Info.plist"
            fi
        fi
    fi
fi

# ------------------------------------------------------------------ summary
printf '\n'
if [ "$FAILURES" -eq 0 ]; then
    printf 'preflight passed (%d warning(s))\n' "$WARNINGS"
    printf 'next: bump versions, archive/notarize/staple, then Tools/sparkle-release.sh\n'
    exit 0
fi
printf 'preflight FAILED: %d gate(s) failed, %d warning(s)\n' "$FAILURES" "$WARNINGS" >&2
printf 'fix the failures above before releasing (%s)\n' "$REPO_SLUG" >&2
exit 1
