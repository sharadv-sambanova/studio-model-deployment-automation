import csv
import json
from utils import STUDIO_INVENTORY_PATH

CLOUD_INVENTORY_PATH="/Users/sharadv/code/studio-model-deployment-automation/inventory/output/cloud_inventory.csv"
CLOUD_ONLY_OUTPUT="output/cloud_only_inventory.csv"
STUDIO_ONLY_OUTPUT="output/studio_only_inventory.csv"
COMMON_OUTPUT="output/common_inventory.csv"

def get_inventories():
    """Parse the inventory files and return maps of key (tuple) -> row (dict)"""

    def studio_filter(row):
        """Only consider rows in Studio inventory that meet these criteria"""
        return row["mode"] == "infer" and  \
        row["rdu_arch"] == "sn40-16" and \
        int(row["model_parallel_rdus"]) == 16 and \
        row["model_app_name"].endswith("Experts")

    # TODO: collapse these into one function
    def studio_row_to_key(row):
        key = (row["model_app_name"], row["param_count"], row["max_seq_length"], row["spec_decoding"])
        key = tuple(str(x).lower() for x in key)
        return key
    def cloud_row_to_key(row):
        key = (row["model_app_name"], row["param_count"], row["max_seq_len"], row["spec_decoding"])
        key = tuple(str(x).lower() for x in key)
        return key

    cloud_inventory, studio_inventory = {}, {}

    with open(STUDIO_INVENTORY_PATH) as f:
        reader = csv.DictReader(f)
        for row in filter(studio_filter, reader):
            studio_inventory[studio_row_to_key(row)] = row
    with open(CLOUD_INVENTORY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cloud_inventory[cloud_row_to_key(row)] = row

    return cloud_inventory, studio_inventory

def compare_inventories(cloud_inventory, studio_inventory):
    """Get the common, cloud_only, and studio_only inventory keys (sets of tuples)"""
    studio_keys = set(studio_inventory.keys())
    cloud_keys = set(cloud_inventory.keys())
    common = studio_keys.intersection(cloud_keys)
    cloud_only = cloud_keys.difference(studio_keys)
    studio_only = studio_keys.difference(cloud_keys)

    return common, cloud_only, studio_only

def write_cloud_only(keys, cloud, studio):
# For each config in cloud inventory that's not in studio inventory
#   It could be a config that matches an app_name + param_count + spec_decoding combination in Studio
#   If so, we need to find the sibling PEFs and a checkpoint that Studio would use for that other combination
#   Else, this is an entirely new config, so we don't have sibling PEFs or Studio checkpoints
#     pass

    fields = [
        "id",
        "model_app_name",
        "experts",
        "param_count",
        "max_seq_len",
        "max_seq_len_cloud", 
        "spec_decoding",
        "batch_sizes",
        "cloud_pefs_json",
        "pefs",
        "copy_pefs",
        "sibling_studio_pefs",
        "studio_model",
    ]

    def find_sibling_artifacts(key):
        sibling_studio_pefs = []
        studio_model = None
        for other_key, row in studio.items():
            if key[:2] == other_key[:2]:
                sibling_studio_pefs.append(row["pef_path"])
                studio_model = row["model_path"]
        return sibling_studio_pefs if sibling_studio_pefs else None, studio_model
    with open(CLOUD_ONLY_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            sibling_studio_pefs, studio_model = find_sibling_artifacts(key)
            row = {}
            for field in fields:
                if field in cloud[key]:
                    row[field] = cloud[key][field]
            row["sibling_studio_pefs"] = sibling_studio_pefs
            row["studio_model"] = studio_model
            writer.writerow(row)


def write_studio_only(keys, studio):
    fields = [
        "model_app_name",
        "param_count",
        "max_seq_length",
        "spec_decoding",
        "batch_sizes",
    ]
    with open(STUDIO_ONLY_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            row = {}
            for field in fields:
                if field in studio[key]:
                    row[field] = studio[key][field]
            writer.writerow(row)


def write_common(keys, cloud, studio):
    fields = [
        "app_name",
        "param_count",
        "max_seq_len",
        "max_seq_len_cloud", 
        "spec_decoding",
        "vocab_size",
        "cloud_only_bs",
        "studio_only_bs",
        "common_bs",
        "cloud_pefs_json",
        "studio_pef",
        "studio_model",
    ]
    with open(COMMON_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            cloud_row, studio_row = cloud[key], studio[key]
            cloud_bs, studio_bs = set(json.loads(cloud_row["batch_sizes"])), set(json.loads(studio_row["batch_sizes"]))
            # Same set of batch sizes, write it to the common inventory
            row = {
                    "app_name": key[0],
                    "param_count": key[1],
                    "max_seq_len": key[2],
                    "max_seq_len_cloud": cloud_row["max_seq_len_cloud"],
                    "spec_decoding": key[3],
                    "vocab_size": studio_row["vocab_size"],
                    "cloud_only_bs": None,
                    "studio_only_bs": None,
                    "common_bs": sorted(list(cloud_bs)),
                    "cloud_pefs_json": cloud_row["cloud_pefs_json"],
                    "studio_pef": studio_row["pef_path"],
                    "studio_model": studio_row["model_path"],
                }
            if cloud_bs != studio_bs:
                cloud_only_bs = cloud_bs.difference(studio_bs)
                studio_only_bs = studio_bs.difference(cloud_bs)
                common_bs = cloud_bs.intersection(studio_bs)
                row["cloud_only_bs"] = sorted(list(cloud_only_bs)) if len(cloud_only_bs) > 0 else None
                row["studio_only_bs"] = sorted(list(studio_only_bs)) if len(studio_only_bs) > 0 else None
                row["common_bs"] = sorted(list(common_bs)) if len(common_bs) > 0 else None
            writer.writerow(row)

if __name__ == "__main__":
    cloud_inventory, studio_inventory = get_inventories()
    common, cloud_only, studio_only = compare_inventories(cloud_inventory, studio_inventory)
    write_common(common, cloud_inventory, studio_inventory)
    write_cloud_only(cloud_only, cloud_inventory, studio_inventory)
    write_studio_only(studio_only, studio_inventory)



