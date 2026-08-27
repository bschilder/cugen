#!/usr/bin/env bash
# Mirror corrected artifacts from R2 to the HF dataset standardmodelbio/cugen.
#
# Runs on a pod because both legs are fast there; routing 100 GB through a
# laptop at ~38 MB/s would take ~1.5 h each way.
#
# WHAT goes to HF and what does not:
#   perchrom  46 GB  -- yes. The artifacts people download and browse.
#   ld_*_cis  ~52 GB -- yes. Small enough, and convenient for the viz project.
#   ld_un/ph  ~1.6 TB -- NO. That is why this moved to R2 in the first place.
#
# NO `set -x`: HF_TOKEN and R2 credentials are both in this environment.
set -uo pipefail
WHAT="${WHAT:?WHAT not set}"        # perchrom | cis
D=/root/mirror; mkdir -p $D
exec > >(tee -a /root/mirror.log) 2>&1
say(){ echo "===== $(date -u +%H:%M:%S) mirror/$WHAT $* ====="; }
fail(){ echo "MIRROR_FAIL $WHAT: $*"; exit 1; }
for v in r2_endpoint r2_access_key r2_secret HF_TOKEN; do
    if [[ -n "${!v:-}" ]]; then echo "  $v present? YES"; else fail "no $v"; fi
done

say "deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl >/dev/null 2>&1
command -v rclone >/dev/null || curl -sSL https://rclone.org/install.sh | bash >/dev/null 2>&1
pip install -q --no-input --break-system-packages "huggingface_hub[hf_transfer]>=0.26" \
    >/dev/null 2>&1 || fail "pip"
export HF_HUB_ENABLE_HF_TRANSFER=1

R2=( --s3-provider Cloudflare --s3-endpoint "$r2_endpoint"
     --s3-access-key-id "$r2_access_key" --s3-secret-access-key "$r2_secret"
     --s3-no-check-bucket )
SRC=":s3:smb-data-prod-scratch/cugen/1kg-30x-grch38"

# upload_large_folder has NO path_in_repo -- it mirrors folder_path to the repo
# ROOT. A first attempt set a DEST variable and used it only in the verify step,
# so 484 files landed at the top level of the dataset. The LOCAL tree must
# therefore be shaped exactly like the wanted repo layout, and ROOT is what gets
# uploaded.
ROOT="$D/tree"
COHORT="1kg-30x-grch38"
case "$WHAT" in
  perchrom)  REMOTE="$SRC/perchrom";  DEST="$COHORT/perchrom" ;;
  cis)       REMOTE="$SRC/ld";        DEST="$COHORT/ld" ;;
  *) fail "WHAT must be perchrom or cis" ;;
esac
LOCAL="$ROOT/$DEST"
mkdir -p "$LOCAL"

say "pull from R2"
if [ "$WHAT" = "cis" ]; then
    # ONLY the cis arms. The genome-wide datasets are ~1.6 TB and stay on R2;
    # an unfiltered pull would fill the disk and defeat the point of moving.
    rclone "${R2[@]}" copy "$REMOTE" "$LOCAL" --include "*_cis/**" \
        --include "provenance.json" --stats-one-line --transfers 16 \
        || fail "pull"
else
    rclone "${R2[@]}" copy "$REMOTE" "$LOCAL" --stats-one-line --transfers 16 \
        || fail "pull"
fi
echo "  pulled $(du -sh "$LOCAL" | cut -f1) in $(find "$LOCAL" -type f | wc -l) files"

say "upload to HF standardmodelbio/cugen"
python3 - <<PYX || fail "hf upload"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "standardmodelbio/cugen"
# upload_large_folder is resumable and parallel; upload_folder would retry the
# whole set on a single failure at this file count.
# folder_path is the TREE ROOT, so the cohort folder becomes the top level.
api.upload_large_folder(repo_id=REPO, repo_type="dataset",
                        folder_path="$ROOT", num_workers=8,
                        print_report=True)
print("  upload complete")
PYX

say "verify: HF file count matches what was pulled"
python3 - <<PYX || fail "verify"
import os, subprocess
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
fs = api.list_repo_files("standardmodelbio/cugen", repo_type="dataset")
have = [f for f in fs if f.startswith("$DEST/")]
local = int(subprocess.run(["bash", "-c", "find $LOCAL -type f | wc -l"],
                           capture_output=True, text=True).stdout.strip())
root_dump = [f for f in fs if "/" not in f and f != ".gitattributes"]
print(f"  local files {local:,}  |  on HF under $DEST/: {len(have):,}")
if root_dump:
    raise SystemExit(f"FILES AT REPO ROOT: {root_dump[:5]} -- layout is wrong")
if len(have) < local:
    raise SystemExit(f"MISSING {local-len(have)} files on HF")
PYX
say "MIRROR_OK $WHAT"
