"""Drive cugen.ld's GPU portability matrix across RunPod GPU types.

For each GPU: create pod -> wait for ssh -> install -> run probe -> collect JSON
-> delete pod. Failures are recorded, never fatal: a GPU that cannot run the
kernel is a RESULT, not an error.

The API key is read from ~/.runpod/config.toml and never printed.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

CFG = os.path.expanduser("~/.runpod/config.toml")
KEY = None
for _line in open(CFG):
    if _line.strip().lower().startswith("apikey"):
        KEY = _line.partition("=")[2].strip().strip('"').strip("'")
        break
assert KEY, "no apikey in ~/.runpod/config.toml"

API = "https://rest.runpod.io/v1"
SSH_KEY = os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
PUBKEYS = open(os.path.expanduser(
    "/private/tmp/claude-501/-Users-bschilder-code-hg-horizon-web/"
    "4ba2546b-e6eb-4c0f-81cc-b69a42be0155/scratchpad/pubkeys.txt")).read().strip()
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DCS = ["US-NY-1", "US-NJ-1", "US-MD-1", "US-PA-2", "US-OH-2", "US-IL-1",
       "US-VA-1", "CA-MTL-3", "US-KS-2", "US-TX-3", "US-OR-1"]

# Volta (2017) through Blackwell (2025); 6 GB through 96 GB.
MATRIX = [
    ("NVIDIA RTX A2000",               "RTX A2000 (6 GB, entry)",     6),
    ("Tesla V100-PCIE-16GB",           "V100 (Volta, 2017)",         16),
    ("NVIDIA RTX 2000 Ada Generation", "RTX 2000 Ada (entry)",       16),
    ("NVIDIA RTX A4000",               "RTX A4000 (Ampere)",         16),
    ("NVIDIA RTX A5000",               "RTX A5000 (Ampere)",         24),
    ("NVIDIA GeForce RTX 3090",        "RTX 3090 (consumer Ampere)", 24),
    ("NVIDIA GeForce RTX 4090",        "RTX 4090 (Ada)",             24),
    ("NVIDIA L40S",                    "L40S (Ada datacenter)",      48),
    ("NVIDIA H100 80GB HBM3",          "H100 SXM (Hopper)",          80),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition",
     "RTX PRO 6000 (Blackwell)", 96),
]


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:400]}


def create(gpu_id, name):
    """Try COMMUNITY (cheaper, where the older GPUs live), then SECURE, then
    without a datacenter restriction. A GPU we cannot obtain is reported as
    unavailable rather than silently skipped."""
    attempts = [("SECURE", DCS), ("COMMUNITY", DCS),
                ("SECURE", None), ("COMMUNITY", None)]
    last = None
    for cloud, dcs in attempts:
        body = {"name": name, "imageName": IMAGE, "gpuTypeIds": [gpu_id],
                "gpuCount": 1, "cloudType": cloud,
                "containerDiskInGb": 60, "ports": ["22/tcp"],
                "env": {"PUBLIC_KEY": PUBKEYS}}
        if dcs:
            body["dataCenterIds"] = dcs
        r = api("POST", "/pods", body)
        if r.get("id"):
            r["_cloud"] = cloud
            return r
        last = r
    return last or {"_error": "all attempts failed"}


def ssh_endpoint(pod_id, tries=30):
    """REST v1 leaves `runtime` null and exposes the mapping at the top level
    as portMappings {"22": <public port>} + publicIp. Reading runtime.ports
    (the v2 shape) makes every pod look like it has no SSH."""
    for _ in range(tries):
        p = api("GET", f"/pods/{pod_id}") or {}
        pm = p.get("portMappings") or {}
        ip = p.get("publicIp")
        if ip and pm.get("22"):
            return ip, pm["22"]
        rt = p.get("runtime") or {}
        for prt in rt.get("ports") or []:
            if prt.get("privatePort") == 22 or prt.get("private") == 22:
                return prt.get("ip"), prt.get("publicPort") or prt.get("public")
        time.sleep(10)
    return None, None


def sh(host, port, cmd, timeout=900):
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ConnectTimeout=10", "-i", SSH_KEY, "-p", str(port),
         f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout)


SETUP = r"""
set -uo pipefail
python3 -m venv /root/venv >/dev/null 2>&1
/root/venv/bin/pip install -q --upgrade pip >/dev/null 2>&1
/root/venv/bin/pip install -q "cupy-cuda12x>=12.0" numpy pandas pyarrow pytest 2>&1 | tail -3
git clone -q --branch ld-matrix https://github.com/bschilder/cugen.git /root/cugen 2>&1 | tail -2
cd /root/cugen && /root/venv/bin/python benchmarks/gpu_matrix.py --out /root/res.json --label "LABEL" 2>&1 | tail -40
"""


def run_one(gpu_id, label, vram):
    rec = {"gpu_id": gpu_id, "label": label, "vram_advertised": vram,
           "ok": False, "stage": "create"}
    t0 = time.time()
    pod = create(gpu_id, f"ldmx-{label.split()[0].lower()}")
    if "_error" in pod or not pod.get("id"):
        rec["stage"] = "create_failed"
        rec["error"] = str(pod)[:300]
        print(f"  [{label}] CREATE FAILED: {str(pod)[:160]}")
        return rec
    pid = pod["id"]
    rec["pod_id"] = pid
    rec["datacenter"] = pod.get("dataCenterId")
    rec["cost_hr"] = pod.get("costPerHr") or pod.get("cost")
    print(f"  [{label}] pod {pid} in {rec['datacenter']}", flush=True)
    try:
        host, port = ssh_endpoint(pid)
        if not host:
            rec["stage"] = "no_ssh"
            print(f"  [{label}] no ssh endpoint")
            return rec
        rec["stage"] = "ssh_wait"
        for _ in range(30):
            if sh(host, port, "echo up", timeout=30).returncode == 0:
                break
            time.sleep(10)
        else:
            rec["stage"] = "ssh_timeout"
            print(f"  [{label}] ssh timeout")
            return rec
        rec["stage"] = "run"
        print(f"  [{label}] running probe ...", flush=True)
        r = sh(host, port, SETUP.replace("LABEL", label), timeout=1800)
        rec["stdout_tail"] = r.stdout[-3000:]
        got = sh(host, port, "cat /root/res.json", timeout=60)
        if got.returncode == 0 and got.stdout.strip():
            rec["probe"] = json.loads(got.stdout)
            rec["ok"] = bool(rec["probe"].get("ok"))
            rec["stage"] = "done"
        else:
            rec["stage"] = "no_result"
            rec["error"] = (r.stdout[-1500:] + r.stderr[-800:])
    except Exception as e:                                    # noqa: BLE001
        rec["stage"] = "exception"
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        api("DELETE", f"/pods/{pid}")
        rec["wall_s"] = round(time.time() - t0, 1)
        print(f"  [{label}] {rec['stage']}  ok={rec['ok']}  "
              f"({rec['wall_s']:.0f}s, pod deleted)", flush=True)
    return rec


def main():
    only = sys.argv[1:] or None
    out = []
    for gpu_id, label, vram in MATRIX:
        if only and not any(o.lower() in label.lower() for o in only):
            continue
        print(f"=== {label} ===", flush=True)
        out.append(run_one(gpu_id, label, vram))
        json.dump(out, open("gpu_matrix_results.json", "w"), indent=2)
    print("\nwrote gpu_matrix_results.json")
    for r in out:
        p = r.get("probe") or {}
        runs = [x for x in p.get("runs", []) if "wall_s" in x]
        best = max(runs, key=lambda x: x["p"]) if runs else None
        print(f"{r['label']:28s} ok={str(r['ok']):5s} "
              f"cc={p.get('compute_capability','?'):>5s} "
              f"{('%.2fs @ p=%d' % (best['wall_s'], best['p'])) if best else r['stage']}")


if __name__ == "__main__":
    main()
