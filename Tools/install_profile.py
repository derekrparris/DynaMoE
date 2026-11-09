#!/usr/bin/env python3
#
# Install a provisioning profile and prove it matches this Mac's keychain.
#
#   Tools/install_profile.py <file.provisionprofile> [--check]
#
# Why this exists: Xcode reports "Provisioning profile X doesn't include signing
# certificate Y" when a profile on disk pins a certificate this Mac no longer
# has. Two certificates can share the same name ("Developer ID Application: Name
# (TEAM)"), so the only reliable comparison is the certificate bytes. This script
# checks the profile's entitlements, its expiry and its certificate against the
# keychain, then installs it and removes older profiles with the same name so
# Xcode cannot pick the stale one.
#
# --check validates and prints, without touching the installed profiles.

import hashlib
import os
import plistlib
import subprocess
import sys
import time

EXPECTED_TEAM = "RMY84Y8G7M"
EXPECTED_APP_ID = "RMY84Y8G7M.DRP.DynaMoE"
# PROVISIONING_PROFILE_SPECIFIER in the project: Xcode resolves a manually
# signed Release profile by this name, so a differently named profile installs
# cleanly and then fails the build.
EXPECTED_PROFILE_NAME = "DynaMoE Developer ID"
# A profile this close to its ExpirationDate is still valid today and lapses
# inside the release you are about to ship.
EXPIRING_SOON_SECONDS = 30 * 24 * 60 * 60
PROFILE_DIRS = [
    os.path.expanduser("~/Library/Developer/Xcode/UserData/Provisioning Profiles"),
    os.path.expanduser("~/Library/MobileDevice/Provisioning Profiles"),
]


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def load_profile(path):
    result = subprocess.run(
        ["security", "cms", "-D", "-i", path], capture_output=True
    )
    if result.returncode != 0 or not result.stdout:
        fail(f"{path} is not a readable provisioning profile")
    try:
        return plistlib.loads(result.stdout)
    except Exception as error:  # noqa: BLE001 - surface the parser's own message
        fail(f"could not parse {path}: {error}")


def identity_hashes():
    """SHA-1 of every code-signing certificate that has a private key."""
    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True,
        text=True,
    )
    hashes = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split('"')
        if len(parts) < 2:
            continue
        token = parts[0].strip().rstrip(")").split(")")[-1].strip()
        name = parts[1]
        if len(token) >= 40:
            hashes[token.upper()] = name
    return hashes


def cert_names_by_sha1(der):
    return hashlib.sha1(der).hexdigest().upper()


def installed_profiles():
    for directory in PROFILE_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".provisionprofile"):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, "rb") as handle:
                    payload = subprocess.run(
                        ["security", "cms", "-D", "-i", path],
                        capture_output=True,
                    )
                if payload.returncode == 0 and payload.stdout:
                    yield path, plistlib.loads(payload.stdout)
            except Exception:  # noqa: BLE001 - an unreadable file is just skipped
                continue


def validate(profile, hashes, now=None):
    """The acceptance rules, as a pure function of the profile and this Mac's
    code-signing identities (digest -> certificate name).

    Returns (problems, notes, matches): `problems` fail the install, `notes` are
    facts worth printing that do not, and `matches` are the Developer ID
    identities the profile pins.
    """
    now = time.time() if now is None else now
    ents = profile.get("Entitlements", {})
    groups = ents.get("keychain-access-groups", [])
    app_id = ents.get("com.apple.application-identifier", "")
    teams = profile.get("TeamIdentifier", [])

    problems = []
    notes = []
    if EXPECTED_TEAM not in teams:
        problems.append(f"team is not {EXPECTED_TEAM}")
    if profile.get("Name") != EXPECTED_PROFILE_NAME:
        notes.append(
            f"profile name is '{profile.get('Name', '(unnamed)')}', but the project resolves "
            f"PROVISIONING_PROFILE_SPECIFIER '{EXPECTED_PROFILE_NAME}'"
        )
    if app_id and app_id != EXPECTED_APP_ID:
        problems.append(f"app id is {app_id}, expected {EXPECTED_APP_ID}")
    elif not app_id:
        # Normal for a Developer ID profile, and the reason the access-group
        # entry below is the check that actually decides this: without an app id
        # there is nothing here to compare, so say that rather than skip quietly.
        notes.append(
            "profile declares no com.apple.application-identifier; "
            "relying on its keychain-access-groups entry"
        )
    # Two values cover this app: the exact group, or the team wildcard. A bare
    # prefix check would also accept RMY84Y8G7M.DRP.OtherApp.
    if not any(g in (EXPECTED_APP_ID, f"{EXPECTED_TEAM}.*") for g in groups):
        problems.append(
            f"no keychain-access-groups entry covering {EXPECTED_APP_ID} "
            f"(accepted: {EXPECTED_APP_ID} or {EXPECTED_TEAM}.*); "
            f"has {', '.join(groups) or 'none'}"
        )

    expires = profile.get("ExpirationDate")
    if expires is None:
        notes.append("profile carries no ExpirationDate; cannot tell how long it lasts")
    else:
        remaining = expires.timestamp() - now
        if remaining <= 0:
            problems.append(
                f"profile expired on {expires:%Y-%m-%d}; it would sign and install, "
                "and the app would then be killed at launch"
            )
        elif remaining <= EXPIRING_SOON_SECONDS:
            notes.append(
                f"profile expires on {expires:%Y-%m-%d}; renew it before the next release"
            )

    dev_id = {
        digest: label
        for digest, label in hashes.items()
        if label.startswith("Developer ID Application")
    }
    if not dev_id:
        problems.append(
            "no 'Developer ID Application' identity with a private key in the keychain"
        )

    certs = profile.get("DeveloperCertificates", [])
    matches = []
    for der in certs:
        digest = cert_names_by_sha1(der)
        # Against dev_id, not every codesigning identity: a profile pinning an
        # Apple Development certificate this Mac holds would otherwise pass and
        # then fail the manually signed Release build.
        if digest in dev_id:
            matches.append(dev_id[digest])
    if not matches:
        pinned = ", ".join(cert_names_by_sha1(der)[:12] for der in certs) or "none"
        pinned_names = [
            hashes[cert_names_by_sha1(der)]
            for der in certs
            if cert_names_by_sha1(der) in hashes
        ]
        have = ", ".join(
            f"{digest[:12]} ({label})" for digest, label in dev_id.items()
        ) or "none"
        hint = (
            " (this Mac does hold " + ", ".join(pinned_names) + ", which is not "
            "a Developer ID Application identity)" if pinned_names else ""
        )
        problems.append(
            "the profile pins certificate(s) " + pinned + " which this Mac has no "
            f"Developer ID private key for{hint} (keychain has {have}); "
            "regenerate the profile against the installed certificate"
        )
    return problems, notes, matches


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv[1:]
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(__doc__ or "usage: Tools/install_profile.py <profile> [--check]")
        sys.exit(0)
    if len(args) != 1:
        print(__doc__ or "usage: Tools/install_profile.py <profile> [--check]", file=sys.stderr)
        sys.exit(2)

    path = os.path.expanduser(args[0])
    if not os.path.isfile(path):
        fail(f"no such file: {path}")

    profile = load_profile(path)
    ents = profile.get("Entitlements", {})

    print(f"profile:    {profile.get('Name', '(unnamed)')}")
    print(f"uuid:       {profile.get('UUID', '(no uuid)')}")
    print(f"app id:     {ents.get('com.apple.application-identifier') or '(none)'}")
    print(f"groups:     {', '.join(ents.get('keychain-access-groups', [])) or '(none)'}")
    print(f"team:       {', '.join(profile.get('TeamIdentifier', [])) or '(none)'}")

    problems, notes, matches = validate(profile, identity_hashes())

    for note in notes:
        print(f"note:      {note}")
    if problems:
        for problem in problems:
            print(f"INVALID:   {problem}")
        sys.exit(1)

    print(f"matches:    {matches[0]}")
    name = profile.get("Name", "(unnamed)")
    uuid = profile.get("UUID", "(no uuid)")

    if check_only:
        print("ok (not installed: --check)")
        return

    targets = []
    for directory in PROFILE_DIRS:
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(directory, f"{uuid}.provisionprofile")
        with open(path, "rb") as source, open(target, "wb") as destination:
            destination.write(source.read())
        targets.append(target)

    removed = []
    for directory in PROFILE_DIRS:
        if not os.path.isdir(directory):
            continue
        for other_path, other in installed_profiles():
            if os.path.dirname(other_path) != directory:
                continue
            if other.get("Name") == name and other.get("UUID") != uuid:
                os.remove(other_path)
                removed.append(other_path)

    for target in targets:
        print(f"installed:  {target}")
    for stale in removed:
        print(f"removed:    {stale} (same name, older certificate)")
    print("next:       rebuild, then check Contents/embedded.provisionprofile in the product")


if __name__ == "__main__":
    main()
