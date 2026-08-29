#!/usr/bin/env bash
# Refreshes the temporary S3 credentials in autotune.yml from this instance's
# IAM role via IMDSv2.  The role's STS session lasts a few hours, and every
# benchmark in test/io/ that takes --config autotune.yml stops working once
# it expires -- this is the one-line fix for that.
#
# Usage:
#   ./refresh_autotune_creds.sh [path/to/autotune.yml]
#
# Env overrides:
#   ROLE       IAM role name (default: S3TestBucketInstanceRole)
#   IMDS_HOST  metadata endpoint (default: 169.254.169.254)

set -euo pipefail

YAML_PATH="${1:-autotune.yml}"
ROLE="${ROLE:-S3TestBucketInstanceRole}"
IMDS_HOST="${IMDS_HOST:-169.254.169.254}"

if [[ ! -f "$YAML_PATH" ]]; then
  echo "refresh_autotune_creds: $YAML_PATH does not exist -- nothing to patch" >&2
  echo "  (create it first; see the header comment this script writes for the shape)" >&2
  exit 1
fi

TOKEN=$(curl -sf -X PUT "http://${IMDS_HOST}/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600") \
  || { echo "refresh_autotune_creds: could not reach IMDSv2 at ${IMDS_HOST} -- not running on EC2?" >&2; exit 1; }

CREDS_JSON=$(curl -sf -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  "http://${IMDS_HOST}/latest/meta-data/iam/security-credentials/${ROLE}") \
  || { echo "refresh_autotune_creds: no credentials for role '${ROLE}' -- check ROLE is right" >&2; exit 1; }

python3 - "$YAML_PATH" "$CREDS_JSON" <<'PYEOF'
import json
import re
import sys

yaml_path, creds_json = sys.argv[1], sys.argv[2]
creds = json.loads(creds_json)

text = open(yaml_path).read()

def replace_field(text, key, value):
    pattern = re.compile(rf"^(\s*{key}:\s*)'[^']*'", re.MULTILINE)
    new_text, n = pattern.subn(rf"\1'{value}'", text, count=1)
    if n == 0:
        raise SystemExit(f"refresh_autotune_creds: no '{key}:' line found in {yaml_path} "
                         "-- is this the expected autotune.yml shape?")
    return new_text

text = replace_field(text, "access_key", creds["AccessKeyId"])
text = replace_field(text, "secret_key", creds["SecretAccessKey"])
text = replace_field(text, "session_token", creds["Token"])

# Keep the "Expire ..." header comment in sync, if present.
text = re.sub(r"(# Credentials generated from IMDSv2, role \S+\. Expire )\S+(\.)",
             rf"\g<1>{creds['Expiration']}\g<2>", text, count=1)

open(yaml_path, "w").write(text)
print(f"refresh_autotune_creds: {yaml_path} updated, expires {creds['Expiration']}")
PYEOF
