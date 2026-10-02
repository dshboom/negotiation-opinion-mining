#!/usr/bin/env python3
import os, sys, time, requests
BASE = "https://www.modelscope.cn/api/v1/models"

def list_files(owner, name, rev="master"):
    r = requests.get(f"{BASE}/{owner}/{name}/repo/files",
                     params={"Revision": rev, "Recursive": "True"}, timeout=60)
    r.raise_for_status()
    return r.json().get("Data", {}).get("Files", [])

def dl_file(owner, name, path, dest, rev="master"):
    url = f"{BASE}/{owner}/{name}/repo"
    r = requests.get(url, params={"Revision": rev, "FilePath": path},
                     stream=True, timeout=300, allow_redirects=True)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    n = 0
    with open(tmp, "wb") as f:
        for c in r.iter_content(1 << 22):
            f.write(c); n += len(c)
    os.replace(tmp, dest)
    return n

def main(mid, outdir):
    owner, name = mid.split("/", 1)
    files = list_files(owner, name)
    blobs = [f for f in files if f.get("Type") == "blob"]
    total = sum(f.get("Size", 0) for f in blobs)
    print(f"[{mid}] {len(blobs)} files, {total/1024**3:.2f} GiB -> {outdir}", flush=True)
    t0 = time.time(); done = 0
    for k, f in enumerate(blobs, 1):
        p = f["Path"]; size = f.get("Size", 0)
        dest = os.path.join(outdir, p)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            done += size
            print(f"  skip {p} ({size/1024**3:.2f}GiB)", flush=True); continue
        n = dl_file(owner, name, p, dest)
        done += n
        el = time.time() - t0
        print(f"  [{k}/{len(blobs)}] {p} {n/1048576:.0f}MB  "
              f"total {done/1024**3:.2f}/{total/1024**3:.2f}GiB  "
              f"{done/1024**2/max(el,1):.1f}MB/s", flush=True)
    print(f"[DONE] {mid} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
