from typing import Dict, List
import subprocess
import json
import csv
import base64
import yaml

from utils import replace_af_prefix, read_csv, STUDIO_INVENTORY_PATH
from cloud_inventory import OUTPUT_FILE as CLOUD_INVENTORY_PATH
    

# cloud name -> studio name mappings
with open("cloud_studio_model_mappings.yaml") as f:
    MODEL_MAPPINGS = yaml.safe_load(f)

def _get_cloud_model_paths(cloud_inventory: List[Dict]):
    """Given the cloud inventory, return a dict of model name -> path"""
    model_paths = {}
    for row in cloud_inventory:
        # Convert the string representation of the dictionary to an actual dictionary
        cloud_models_dict = json.loads(row['cloud_models'].replace("'",'"'))
        model_paths.update(cloud_models_dict)

    return model_paths
        

def _get_studio_model_paths(studio_inventory: List[Dict]):
    """Given the studio inventory, return a dict of model name -> path"""
    model_paths = {}
    for row in studio_inventory:
        model_name = row['model_checkpoint_name']
        model_paths[model_name] = row['model_path']
    return model_paths

def _compare_hashes(cloud_hashes, studio_hashes):
    """Compare the hashes of files with the same name from two GCS paths."""
    common_files = set(cloud_hashes.keys()) & set(studio_hashes.keys())
    different_hashes = {}
    for file_name in common_files:
        if cloud_hashes[file_name] != studio_hashes[file_name]:
            different_hashes[file_name] = {
                'cloud': cloud_hashes[file_name],
                'studio': studio_hashes[file_name]
            }
    cloud_only = set(cloud_hashes.keys()) - set(studio_hashes.keys())
    studio_only = set(studio_hashes.keys()) - set(cloud_hashes.keys())
    return different_hashes, cloud_only, studio_only

def _get_hashes_gcs(path):
    """Get md5sums for all files under <path> using gsutil ls -L
    gsutil ls -L output looks like this:
    gs://acp-coe-models-checkpoints-prod-0/version/0.1.0/pefs-checkpoints/ckpts/Llama-Guard-3-8B/.gitattributes:
        Creation time:          Thu, 14 Nov 2024 22:00:18 GMT
        Update time:            Thu, 14 Nov 2024 22:00:18 GMT
        Storage class:          STANDARD
        Content-Length:         1519
        Content-Type:           application/octet-stream
        Metadata:               
            goog-reserved-file-mtime:1731487159
        Hash (crc32c):          AjbvVQ==
        Hash (md5):             qFn4qJaFdH/9QXG4cFQMQQ==
    <next file>
    """
    file_hashes = {}
    command = f"gsutil ls -L {path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stderr:
        raise subprocess.SubprocessError(f"Error message: {result.stderr}")
    
    lines = result.stdout.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("gs://"):
            file_name = line.split('/')[-1].rstrip(':') # Just the basename
        if line.startswith("Hash (md5):"):
            md5sum_encoded = line.split()[-1]
            md5sum = base64.b64decode(md5sum_encoded).hex()
            file_hashes[file_name] = md5sum
    return file_hashes

def _get_hashes_af(af_path):
    """Given an artifactory folder path, return a dict of md5sums for all files under that path"""
    hashes = {}
    af_path = replace_af_prefix(af_path)
    command = f"jf rt s {af_path}"

    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert output.returncode == 0, f"Got bad returncode for '{command}' with error {output.stderr}"
    output = output.stdout

    out_json = json.loads(output)
    # out_json is a list of file metadata for each file in the coe_pef folder
    for file_metadata in out_json:
        file_name = file_metadata["path"].split("/")[-1]
        hashes[file_name] = file_metadata["md5"]
    
    return hashes

def _compare_paths(cloud_path, studio_path):
    """Compare two folders cloud_path and studio_path for equality of all files"""
    cloud_hashes = _get_hashes_gcs(cloud_path)
    if studio_path.startswith("gs://"):
        studio_hashes =  _get_hashes_gcs(studio_path)
    else:
        studio_hashes = _get_hashes_af(studio_path)
    return _compare_hashes(cloud_hashes, studio_hashes)        

def compare_models(cloud_inventory, studio_inventory):
    cloud_models = _get_cloud_model_paths(cloud_inventory)
    studio_models = _get_studio_model_paths(studio_inventory)

    rows = []

    # In addition to the explicit mappings in MODEL_MAPPINGS, 
    # we also want to include models that have the same name in both inventories
    same_name = [m for m in cloud_models if m in studio_models]
    MODEL_MAPPINGS.update({m:m for m in same_name})

    for cloud_name, studio_name in MODEL_MAPPINGS.items():
        cloud_path = cloud_models[cloud_name]
        studio_path = studio_models[studio_name]

        # Check if the models from MODEL_MAPPINGS exist in the inventories
        if not cloud_name in cloud_models:
            raise ValueError(f"{cloud_name} not found in cloud models")
        if not studio_name in studio_models:
            raise ValueError(f"{studio_name} not found in studio models")
        
        differing_files, cloud_only, studio_only = _compare_paths(cloud_path, studio_path)
        row = {
            'cloud_model_name': cloud_name,
            'studio_model_name': studio_name,
            'cloud_path': cloud_path,
            'studio_path': studio_path,
            'is_same': len(differing_files) == 0 and len(cloud_only) == 0 and len(studio_only) == 0,
            'differing_files': sorted(list(differing_files.keys())),
            'cloud_only_files': sorted(list(cloud_only)),
            'studio_only_files': sorted(list(studio_only))
        }
        rows.append(row)
    return rows

# if __name__ == "__main__":
#     cloud_inventory = read_csv(CLOUD_INVENTORY_PATH)
#     studio_inventory = read_csv(STUDIO_INVENTORY_PATH)
#     compare_models(cloud_inventory, studio_inventory, "test.csv")
