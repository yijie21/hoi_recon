"""HOI4D official OneDrive downloader (anonymous badger-token flow).

The hoi4d.github.io share links were migrated to SharePoint, so the classic
api.onedrive.com anonymous API fails; this uses the same anonymous token the
OneDrive web UI uses. Download URLs are pre-signed and expire (~1 h), so each
file re-resolves its URL right before download. Resumable (curl -C -).

Usage: python hoi4d_download.py [component ...]
Components: rgb_video cad_models annotations camera hand_pose depth_video
Default: all. Target: /workspace/datasets/hoi4d/
"""
import base64
import json
import os
import subprocess
import sys
import time

DEST = "/workspace/datasets/hoi4d"
LINKS = {
 "rgb_video":  "https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIvAQAAAAAB4AcDOxj_uuh7alRaR9b7MQ?e=GyBNaD",
 "cad_models": "https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIsAQAAAAAB5gBbXuK7eDzXQmzDS5W09g?e=DdkFTE",
 "annotations":"https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIuAQAAAAABELgpzxDxCz58Qov76-wpvw?e=qhyY0h",
 "camera":     "https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIqAQAAAAABZBWQ3p5gxd-tu_3rOeSo_A?e=GjWeg8",
 "hand_pose":  "https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIrAQAAAAABHyHqDhs5CpJAkWTAQoGWxQ?e=ALFMCD",
 "depth_video":"https://1drv.ms/f/c/12e5c3dbeffd0594/EpQF_e_bw-UggBIpAQAAAAABswaE_tZsrSwsEoBbFh7F2w?e=K797bs",
}
API = "https://my.microsoftpersonalcontent.com/_api/v2.0"


def curl_json(url, tok):
    out = subprocess.run(
        ["curl", "-s", "-m", "120", url,
         "-H", f"Authorization: Badger {tok}", "-H", "Prefer: autoredeem"],
        capture_output=True, text=True).stdout
    return json.loads(out)


def token():
    out = subprocess.run(
        ["curl", "-s", "-m", "60", "-X", "POST",
         "https://api-badgerp.svc.ms/v1.0/token",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"appId": "5cbed6ac-a083-4e14-b191-b4ba07653de2"})],
        capture_output=True, text=True).stdout
    return json.loads(out)["token"]


def share_id(url):
    return "u!" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


def fetch_file(dl_url, dest, size):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"    OK (cached) {os.path.basename(dest)}", flush=True)
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    rc = subprocess.run(["curl", "-L", "-C", "-", "--retry", "5",
                         "--retry-delay", "10", "-o", dest, "-s", dl_url]).returncode
    got = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f"    {'OK' if got == size else 'SHORT'} {os.path.basename(dest)} "
          f"{got/1e9:.2f}/{size/1e9:.2f} GB (rc={rc})", flush=True)
    return got == size


def walk_folder(drive, item, tok, rel, out):
    url = f"{API}/drives/{drive}/items/{item}/children?$top=500"
    while url:
        d = curl_json(url, tok)
        for c in d.get("value", []):
            p = os.path.join(rel, c["name"])
            if "folder" in c:
                walk_folder(drive, c["id"], tok, p, out)
            else:
                out.append((c["id"], p, c["size"]))
        url = d.get("@odata.nextLink")


def main():
    names = sys.argv[1:] or list(LINKS)
    os.makedirs(DEST, exist_ok=True)
    for name in names:
        print(f"== {name} ==", flush=True)
        for attempt in range(10):
            try:
                tok = token()
                d = curl_json(f"{API}/shares/{share_id(LINKS[name])}/driveitem", tok)
                if "folder" in d:
                    drive = d["parentReference"]["driveId"]
                    files = []
                    walk_folder(drive, d["id"], tok, d["name"], files)
                    print(f"  folder: {len(files)} files, "
                          f"{sum(s for _, _, s in files)/1e9:.1f} GB", flush=True)
                    done = True
                    for fid, rel, size in files:
                        dest = os.path.join(DEST, rel)
                        if os.path.exists(dest) and os.path.getsize(dest) == size:
                            continue
                        item = curl_json(f"{API}/drives/{drive}/items/{fid}", token())
                        ok = fetch_file(item["@content.downloadUrl"], dest, size)
                        done = done and ok
                else:
                    done = fetch_file(d["@content.downloadUrl"],
                                      os.path.join(DEST, d["name"]), d["size"])
                if done:
                    break
                print(f"  retrying {name} (attempt {attempt + 2})", flush=True)
                time.sleep(15)
            except Exception as e:
                print(f"  error on {name}: {e}; retrying", flush=True)
                time.sleep(30)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
