import os
import sys
import json
import hashlib
import subprocess
import tempfile
import time
import requests
from packaging.version import Version
from app.version import __version__

GITHUB_OWNER = "michalneoral"              # <-- change
GITHUB_REPO  = "test_kinetika"                 # <-- change
ASSET_PREFIX = "myapp-setup-"          # e.g. myapp-setup-1.2.3.exe
TIMEOUT_S    = 10

def get_latest_release():
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    r = requests.get(url, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()

def pick_windows_installer_asset(release):
    # Expect an asset named like myapp-setup-<version>.exe and optional checksums.txt
    exe = None
    checksums = None
    for a in release.get("assets", []):
        name = a["name"]
        if name.endswith(".exe") and name.startswith(ASSET_PREFIX):
            exe = a
        elif name.lower().startswith("checksums") and name.lower().endswith(".txt"):
            checksums = a
    return exe, checksums

def download_asset(asset, dst_path):
    url = asset["browser_download_url"]
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)

def parse_checksums_file(text):
    # expect lines like: SHA256  <hash>  <filename>
    out = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            algo, digest, filename = parts[0], parts[1], " ".join(parts[2:])
            if algo.upper().startswith("SHA256"):
                out[filename] = digest.lower()
    return out

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest().lower()

def run_installer_silently(installer_path):
    """
    Inno Setup supports silent flags:
      /VERYSILENT /NORESTART /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS
    """
    args = [
        installer_path,
        "/VERYSILENT",
        "/CLOSEAPPLICATIONS",
        "/RESTARTAPPLICATIONS",
    ]
    # spawn detached so the installer can elevate if needed
    creationflags = 0x00000008  # CREATE_NEW_CONSOLE
    try:
        subprocess.Popen(args, creationflags=creationflags)
        return True
    except Exception:
        return False

def maybe_update_in_background():
    try:
        release = get_latest_release()
        tag = release.get("tag_name", "").lstrip("v")
        if not tag:
            return
        current = Version(__version__)
        latest = Version(tag)
        print('Current version:', current)
        print('Latest version:', latest)
        if latest <= current:
            print('App is up to date.')
            return  # up to date

        exe_asset, checks_asset = pick_windows_installer_asset(release)
        if not exe_asset:
            return

        with tempfile.TemporaryDirectory() as td:
            exe_name = exe_asset["name"]
            exe_path = os.path.join(td, exe_name)
            download_asset(exe_asset, exe_path)

            # optional checksum verification
            if checks_asset:
                checks_txt = os.path.join(td, checks_asset["name"])
                download_asset(checks_asset, checks_txt)
                checks = parse_checksums_file(open(checks_txt, "r", encoding="utf-8", errors="ignore").read())
                expected = checks.get(exe_name)
                if expected:
                    got = sha256_file(exe_path)
                    if got != expected:
                        print("Checksum mismatch, aborting update.")
                        return

            # Launch installer, exit app to allow replacement
            ok = run_installer_silently(exe_path)
            if ok:
                # give the installer a head start, then quit this process
                time.sleep(1.0)
                os._exit(0)
    except Exception as e:
        # swallow errors to avoid breaking the app
        print(f"Update check failed: {e}")
