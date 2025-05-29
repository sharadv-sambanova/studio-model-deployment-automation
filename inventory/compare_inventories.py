import csv
import json
from typing import Dict, Set, List

from utils import STUDIO_INVENTORY_PATH, convert_seq_len
from schemas import InventoryKey
from compare_pefs import compare_pefs
from cloud_inventory import OUTPUT_FILE as CLOUD_INVENTORY_PATH

CLOUD_ONLY_OUTPUT="output/cloud_only_inventory.csv"
STUDIO_ONLY_OUTPUT="output/studio_only_inventory.csv"
COMMON_OUTPUT="output/common_inventory.csv"
ONBOARD_TO_STUDIO_OUTPUT="output/onboard_to_studio.csv"

def get_inventories():
    """Parse the inventory files and return maps of key (tuple) -> row (dict)"""

    def studio_filter(row):
        """Only consider rows in Studio inventory that meet these criteria"""
        return row["mode"] == "infer" and  \
        row["rdu_arch"] == "sn40-16" and \
        int(row["model_parallel_rdus"]) == 16 and \
        row["model_app_name"].endswith("Experts")
    
    cloud_inventory, studio_inventory = {}, {}

    with open(STUDIO_INVENTORY_PATH) as f:
        reader = csv.DictReader(f)
        for row in filter(studio_filter, reader):
            studio_inventory[InventoryKey.from_input(row)] = row
    with open(CLOUD_INVENTORY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cloud_inventory[InventoryKey.from_input(row)] = row

    return cloud_inventory, studio_inventory

class InventoryComparer():

    def __init__(self):
        self.cloud_inventory, self.studio_inventory = get_inventories()
        self.common, self.cloud_only, self.studio_only = self._compare_inventory_keys()

    def _compare_inventory_keys(self):
        """Get the common, cloud_only, and studio_only inventory keys (sets of InventoryKey)"""
        studio_keys = set(studio_inventory.keys())
        cloud_keys = set(cloud_inventory.keys())
        
        common = studio_keys.intersection(cloud_keys)
        cloud_only = cloud_keys.difference(studio_keys)
        studio_only = studio_keys.difference(cloud_keys)

        return common, cloud_only, studio_only
    
    def _write_file(self, rows, fields, filename):
        with open(filename, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write(self):
        pass


def compare_inventory_keys(cloud_inventory, studio_inventory):
    """Get the common, cloud_only, and studio_only inventory keys (sets of InventoryKey)"""
    studio_keys = set(studio_inventory.keys())
    cloud_keys = set(cloud_inventory.keys())
    
    common = studio_keys.intersection(cloud_keys)
    cloud_only = cloud_keys.difference(studio_keys)
    studio_only = studio_keys.difference(cloud_keys)

    return common, cloud_only, studio_only

def write_cloud_only(keys: Set[InventoryKey], cloud: Dict[InventoryKey, Dict], studio: Dict[InventoryKey, Dict]):
    fields = [
        "id",
        "group_id",
        "model_app_name",
        "experts",
        "param_count",
        "max_seq_length",
        "max_seq_length_cloud", 
        "spec_decoding",
        "batch_sizes",
        "cloud_pefs_json",
        "cloud_models",
        "sibling_studio_pefs",
        "studio_model",
    ]
    rows = []
    def find_sibling_artifacts(key: InventoryKey):
        sibling_studio_pefs = {}
        studio_model = None
        for other_key, row in studio.items():
            if key.is_sibling(other_key):
                seq_len_key = convert_seq_len(other_key.max_seq_length, str)
                sibling_studio_pefs[seq_len_key] = row["pef_path"].replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
                studio_model = row["model_path"].replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
        return sibling_studio_pefs if sibling_studio_pefs else None, studio_model        
    
    with open(CLOUD_ONLY_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(keys, key= lambda x: str(x)):
            sibling_studio_pefs, studio_model = find_sibling_artifacts(key)
            row = {}
            for field in fields:
                if field in cloud[key]:
                    row[field] = cloud[key][field]
            row["sibling_studio_pefs"] = sibling_studio_pefs
            row["studio_model"] = studio_model
            rows.append(row)

        writer.writerows(rows)
    return rows


def write_onboard_to_studio(common_rows: List[Dict], cloud_only_rows: List[Dict]):
    """
        For each row in common
            if the row has nonmatching pefs OR if the 
            add a row for the cloud batch sizes that don't match
    """
    fields = [
        "id",
        "group_id",
        "model_app_name",
        "experts",
        "param_count",
        "max_seq_length",
        "max_seq_length_cloud", 
        "vocab_size",
        "spec_decoding",
        "batch_sizes",
        "studio_batch_sizes",
        "cloud_pefs_json",
        "cloud_models",
        "studio_model",
        "studio_pef",
        "sibling_studio_pefs",
        "onboard_cloud_models",
        "is_new_config",
    ]
    rows = []
    with open(ONBOARD_TO_STUDIO_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for common_row in common_rows:
            # Only need to onboard rows in common where there are cloud-only BS PEFs or different PEFs for same BS
            if common_row["cloud_only_bs"] == [] and common_row["common_bs_different_pefs"] == []:
                continue

            row = {}
            for field in fields:
                if field in common_row:
                    row[field] = common_row[field]
            row["onboard_cloud_models"] = row["studio_model"] is None
            row["is_new_config"] = False
            rows.append(row)
        for cloud_only_row in cloud_only_rows:
            row = {}
            for field in fields:
                if field in cloud_only_row:
                    row[field] = cloud_only_row[field]
            row["onboard_cloud_models"] = row["studio_model"] is None
            row["is_new_config"] = True
            rows.append(row) 
        writer.writerows(rows)
    return rows


def write_common(keys: Set[InventoryKey], cloud: Dict[InventoryKey, Dict], studio: Dict[InventoryKey, Dict]):

    def find_sibling_artifacts(key: InventoryKey):
        sibling_studio_pefs = {}
        studio_model = None
        for other_key, row in studio.items():
            if key.is_sibling(other_key):
                seq_len_key = convert_seq_len(other_key.max_seq_length, str)
                sibling_studio_pefs[seq_len_key] = row["pef_path"].replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
                studio_model = row["model_path"].replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
        return sibling_studio_pefs if sibling_studio_pefs else None, studio_model   

    fields = [
        "id",
        "group_id",
        "model_app_name",
        "experts",
        "param_count",
        "max_seq_length",
        "max_seq_length_cloud", 
        "vocab_size",
        "spec_decoding",
        "batch_sizes",
        "studio_batch_sizes",
        "cloud_pefs_json",
        "cloud_models",
        "studio_model",
        "studio_pef",
        "sibling_studio_pefs",
        "studio_only_bs",
        "cloud_only_bs",
        "common_bs",
        "common_bs_matching_pefs",
        "common_bs_different_pefs",
        "common_bs_different_pefs_json",
        "date_difference_for_nonmatching_pefs",
    ]
    rows = []
    with open(COMMON_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(keys, key= lambda x: str(x)):
            cloud_row, studio_row = cloud[key], studio[key]
            cloud_bs, studio_bs = set(json.loads(cloud_row["batch_sizes"])), set(json.loads(studio_row["batch_sizes"]))
            row = {}
            for field in fields:
                if field in studio_row:
                    row[field] = studio_row[field]
                if field in cloud_row:
                    row[field] = cloud_row[field]
                row["studio_model"] = studio_row["model_path"]
                row["studio_pef"] = studio_row["pef_path"]
                row["studio_batch_sizes"] = studio_row["batch_sizes"]
            sibling_studio_pefs, _ = find_sibling_artifacts(key)
            row["sibling_studio_pefs"] = sibling_studio_pefs
            # row = {
            #         "id": str(key),
            #         "app_name": key.app_name,
            #         "param_count": key.param_count,
            #         "max_seq_length": key.max_seq_length,
            #         "max_seq_length_cloud": cloud_row["max_seq_length_cloud"],
            #         "spec_decoding": key.sd,
            #         "vocab_size": studio_row["vocab_size"],
            #         "cloud_pefs_json": cloud_row["cloud_pefs_json"],
            #         "cloud_models": cloud_row["cloud_models"],
            #         "studio_pef": studio_row["pef_path"],
            #         "studio_model": studio_row["model_path"],
            #     }
            common_bs = sorted(list(cloud_bs.intersection(studio_bs)))
            cloud_only_bs = sorted(list(cloud_bs.difference(studio_bs)))
            studio_only_bs = sorted(list(studio_bs.difference(cloud_bs)))
            common_bs_with_matching_pefs, common_bs_different_pefs, date_difference = compare_pefs(json.loads(cloud_row["cloud_pefs_json"]), studio_row["pef_path"], common_bs)
            row["studio_only_bs"] = studio_only_bs
            row["cloud_only_bs"] = cloud_only_bs
            row["common_bs"] = common_bs
            row["common_bs_matching_pefs"] = common_bs_with_matching_pefs
            row["common_bs_different_pefs"] = sorted(list(date_difference.keys()))
            row["common_bs_different_pefs_json"] = common_bs_different_pefs

            row["date_difference_for_nonmatching_pefs"] = date_difference
            rows.append(row)

        writer.writerows(rows)
    return rows


def write_studio_only(keys: Set[InventoryKey], studio: Dict[InventoryKey, Dict]):
    fields = [
        "model_app_name",
        "param_count",
        "max_seq_length",
        "spec_decoding",
        "batch_sizes",
    ]
    rows = []
    with open(STUDIO_ONLY_OUTPUT, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(keys, key= lambda x: str(x)):
            row = {}
            for field in fields:
                if field in studio[key]:
                    row[field] = studio[key][field]
            rows.append(row)
        writer.writerows(rows)
    
    return rows


if __name__ == "__main__":
    cloud_inventory, studio_inventory = get_inventories()
    common, cloud_only, studio_only = compare_inventory_keys(cloud_inventory, studio_inventory)
    common_rows = write_common(common, cloud_inventory, studio_inventory)
    cloud_only_rows = write_cloud_only(cloud_only, cloud_inventory, studio_inventory)
    studio_only_rows = write_studio_only(studio_only, studio_inventory)
    onboard_rows = write_onboard_to_studio(common_rows, cloud_only_rows)
