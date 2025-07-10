import subprocess
import json
import yaml
import base64
import dateutil.parser
from datetime import timezone
import os
from pathlib import Path
import atexit
from typing import Dict, List


CACHE_FILE = Path(__file__).parent / ".md5sum_cache.yaml"
with open(CACHE_FILE) as f:
    CACHE = yaml.safe_load(f)
    if CACHE is None:
        CACHE = {}

def write_cache():
    print("Writing cache...", end=" ")
    with open(CACHE_FILE, "w") as f:
        yaml.dump(CACHE, f)
    print("done")

# Write the updated cache before exiting
atexit.register(write_cache)

def cache_metadata(fn):
    """
        Decorator for caching PEF metadata to speed up metadata retrieval
        Checks cache for metadata, then if cache miss runs retrieval function and caches the result
    """
    def check_cache(path: str):
        return CACHE.get(path, None)

    def update_cache(path: str, metadata: Dict):
        CACHE[path] = metadata

    def wrapper(pef_path: str):
        print(f"Checking cache for {pef_path}...", end=" ")
        cached_val = check_cache(pef_path)
        if cached_val is not None:
            print("HIT")
            return cached_val
        print("MISS")
        metadata = fn(pef_path)
        update_cache(pef_path, metadata)
        return metadata

    return wrapper

@cache_metadata
def get_studio_pef_metadata(pef_path: str):
    """
        Retrieve pef metadata from JFrog.
        Parses the data from the output of jf rt s <pef_path>. See below comment for an example of output
    """
    # [
    #   {
    #     "path": "sw-generic-daas-artifacts-dev/inference-engine/2025/pefs/deepseek-r1-16k-fp8-pef-v3/bs1/coe_pef/sncprof.json.gz",
    #     "type": "file",
    #     "size": 11590862,
    #     "created": "2025-03-26T13:53:39.342-07:00",
    #     "modified": "2025-03-26T13:53:39.221-07:00",
    #     "sha1": "676cc424ff69a54a2bea4135176085ae275250da",
    #     "sha256": "c64a33338be0b6bc5219b387ae76998f8d8017d99b01823b1a6e16e0a9928868",
    #     "md5": "a2334cef8b358cc35f3b96b30b13509e"
    #   },
    # ...

    command = f"jf rt s {pef_path}"

    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert output.returncode == 0, f"Got bad returncode for '{command}' with error {output.stderr}"
    output = output.stdout

    out_json = json.loads(output)
    # out_json is a list of file metadata for each file in the coe_pef folder
    for file_metadata in out_json:
        filepath = file_metadata["path"]
        # want the metadata for the .pef file specifically
        if filepath.endswith(".pef"):
            return {
                "md5": file_metadata["md5"], 
                "upload_date": file_metadata["created"], 
                "path": filepath
            }
    
    raise ValueError(f"No .pef file found in output {out_json}")


@cache_metadata
def get_cloud_pef_metadata(pef_path: str):
    """
        Retrieves PEF metadata (md5sum, upload date, path) from GCS.
        Parses data from the output of gsutil stat <filepath>. See comment for an example output
    """

    # <path>:
    #     Creation time:          Thu, 20 Feb 2025 18:27:17 GMT
    #     Update time:            Thu, 20 Feb 2025 18:27:17 GMT
    #     Storage class:          STANDARD
    #     Content-Length:         3354423528
    #     Content-Type:           application/octet-stream
    #     Metadata:               
    #         goog-reserved-file-mtime:1738202406
    #     Hash (crc32c):          YeC9gg==
    #     Hash (md5):             skySRE+gMILYnhNWuLR0Eg==
    #     ETag:                   CKuVxbDw0osDEAE=
    #     Generation:             1740076036999851
    #     Metageneration:         1

    command = f"gsutil stat {pef_path}"
    output = subprocess.run(command.split(), capture_output=True, text=True)
    assert output.returncode == 0, f"Got bad returncode for '{command}' with error {output.stderr}"
    output = output.stdout

    data = {}
    lines = output.split("\n")
    data["path"] = lines[0].strip(":\n")
    for line in lines[1:]:
        line = line.strip()
        key = line.split(":")[0]
        val = ":".join(line.split(":")[1:]).strip()
        data[key] = val
    data["md5_decoded"] = base64.b64decode(data["Hash (md5)"]).hex()
    return {
        "md5": data["md5_decoded"], 
        "upload_date": data["Creation time"], 
        "path": data["path"]
    }


def date_difference(date1, date2):
    """Compute the difference in days between date1 and date2"""
    # Parse the dates using dateutil (handles both formats well)
    dt1 = dateutil.parser.parse(date1)
    dt2 = dateutil.parser.parse(date2)

    # Convert both to UTC (to ensure correct diff)
    dt1_utc = dt1.astimezone(timezone.utc).replace(tzinfo=None)
    dt2_utc = dt2.astimezone(timezone.utc).replace(tzinfo=None)

    # Calculate difference
    delta = dt1_utc - dt2_utc
    return delta.days 

def compare_pefs(cloud_pefs: Dict[str, Dict], studio_pef: str, common_bs: List[int]):
    common_bs_with_matching_pefs, common_bs_different_pefs = [], []
    studio_pef_folder = studio_pef.replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
    for bs in common_bs:
        cloud_pef = cloud_pefs[str(bs)]['pef_path']

        # Studio path contains all the bs pefs, just look at the bs in question
        studio_pef = os.path.join(studio_pef_folder, f"bs{bs}/coe_pef/")
        print(f"Comparing...\n{cloud_pef}\n{studio_pef}")
        
        studio_metadata = get_studio_pef_metadata(studio_pef)
        cloud_metadata = get_cloud_pef_metadata(cloud_pef)

        if studio_metadata["md5"] != cloud_metadata["md5"]:
            print(f"NOT A MATCH")
            bs_json = {
                "batch_size": bs, 
                "cloud_pef": cloud_metadata, 
                "studio_pef": studio_metadata,
                "upload_date_difference_in_days": date_difference(cloud_metadata["upload_date"], studio_metadata["upload_date"])
            }
            common_bs_different_pefs.append(bs_json)
        else:
            print("MATCH")
            common_bs_with_matching_pefs.append(bs)

    return common_bs_with_matching_pefs, common_bs_different_pefs, {x["batch_size"]: x["upload_date_difference_in_days"] for x in common_bs_different_pefs}
