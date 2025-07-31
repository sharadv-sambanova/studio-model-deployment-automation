import csv
import json
from typing import Dict, Set, List, Tuple, Union

from utils import STUDIO_INVENTORY_PATH, convert_seq_len, replace_af_prefix, read_csv
from schemas import InventoryKey
from compare_pefs import compare_pefs
from cloud_inventory import OUTPUT_FILE as CLOUD_INVENTORY_PATH
from compare_models import compare_models

CLOUD_ONLY_OUTPUT="output/cloud_only_inventory.csv"
STUDIO_ONLY_OUTPUT="output/studio_only_inventory.csv"
COMMON_OUTPUT="output/common_inventory.csv"
ONBOARD_TO_STUDIO_OUTPUT="output/onboard_to_studio.csv"
MODEL_COMPARISON_OUTPUT="output/model_comparison.csv"

def get_inventories() -> Tuple[Dict[InventoryKey, Dict], Dict[InventoryKey, Dict]]:
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
    studio_only_fields = [
        "id",
        "group_id",
        "model_app_name",
        "param_count",
        "max_seq_length",
        "spec_decoding",
        "batch_sizes",
    ]
    cloud_only_fields = studio_only_fields + [
        "max_seq_length_cloud", 
        "experts",
        "cloud_pefs_json",
        "cloud_models",
        "sibling_studio_pefs",
        "studio_model",
    ]
    common_fields = cloud_only_fields + [
        "vocab_size",
        "studio_batch_sizes",
        "studio_pef",
        "studio_only_bs",
        "cloud_only_bs",
        "common_bs",
        "common_bs_matching_pefs",
        "common_bs_different_pefs",
        "common_bs_different_pefs_json",
        "date_difference_for_nonmatching_pefs",
    ]
    onboard_to_studio_fields = cloud_only_fields + [
        "vocab_size",
        "studio_batch_sizes",
        "studio_pef",
        "onboard_cloud_models",
        "is_new_config",
    ]
    model_comparison_fields = [
        'cloud_model_name',
        'studio_model_name',
        'cloud_path',
        'studio_path',
        'is_same',
        'differing_files',
        'cloud_only_files',
        'studio_only_files'
    ]

    def __init__(self):
        self.cloud_inventory, self.studio_inventory = get_inventories()
        self.common_keys, self.cloud_only_keys, self.studio_only_keys = self._compare_inventory_keys()
        # raw rows from cloud and studio inventories
        self.cloud_inventory_raw = read_csv(CLOUD_INVENTORY_PATH)
        self.studio_inventory_raw = read_csv(STUDIO_INVENTORY_PATH)


    def _compare_inventory_keys(self) -> Tuple[Set, Set, Set]:
        """Get the common, cloud_only, and studio_only inventory keys (sets of InventoryKey)"""
        studio_keys = set(self.studio_inventory.keys())
        cloud_keys = set(self.cloud_inventory.keys())
        
        common = studio_keys.intersection(cloud_keys)
        cloud_only = cloud_keys.difference(studio_keys)
        studio_only = studio_keys.difference(cloud_keys)

        return common, cloud_only, studio_only
    

    def write(self):
        InventoryComparer._write_file(self._cloud_only_rows(), InventoryComparer.cloud_only_fields, CLOUD_ONLY_OUTPUT)
        InventoryComparer._write_file(self._common_rows(), InventoryComparer.common_fields, COMMON_OUTPUT)
        InventoryComparer._write_file(self._studio_only_rows(), InventoryComparer.studio_only_fields, STUDIO_ONLY_OUTPUT)
        InventoryComparer._write_file(self._onboard_to_studio_rows(), InventoryComparer.onboard_to_studio_fields, ONBOARD_TO_STUDIO_OUTPUT)
        model_comparison_rows = compare_models(self.cloud_inventory_raw, self.studio_inventory_raw)
        InventoryComparer._write_file(model_comparison_rows, self.model_comparison_fields, MODEL_COMPARISON_OUTPUT)


    @staticmethod
    def _write_file(rows, fields, filename):
        # model comparison rows don't have 'id' column, all others do
        rows = sorted(rows, key=lambda x: x["id"] if "id" in x else x["cloud_model_name"])
        with open(filename, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)    


    @staticmethod
    def _compare_rows(cloud_row: Dict, studio_row: Dict) -> Dict:
        cloud_bs, studio_bs = set(json.loads(cloud_row["batch_sizes"])), set(json.loads(studio_row["batch_sizes"]))
        common_bs = sorted(list(cloud_bs.intersection(studio_bs)))
        cloud_only_bs = sorted(list(cloud_bs.difference(studio_bs)))
        studio_only_bs = sorted(list(studio_bs.difference(cloud_bs)))
        common_bs_with_matching_pefs, common_bs_different_pefs, date_difference = compare_pefs(json.loads(cloud_row["cloud_pefs_json"]), studio_row["pef_path"], common_bs)
        return {
            "studio_only_bs": studio_only_bs,
            "cloud_only_bs": cloud_only_bs,
            "common_bs": common_bs,
            "common_bs_matching_pefs": common_bs_with_matching_pefs,
            "common_bs_different_pefs": sorted(list(date_difference.keys())),
            "common_bs_different_pefs_json": common_bs_different_pefs,
            "date_difference_for_nonmatching_pefs": date_difference,
        }
    

    def _find_sibling_artifacts(self, key: InventoryKey) -> Tuple[Union[Dict, None], Union[str, None]]:
        sibling_studio_pefs = {}
        studio_model = None
        for other_key, row in self.studio_inventory.items():
            if key.is_sibling(other_key):
                seq_len_key = convert_seq_len(other_key.max_seq_length, str)
                sibling_studio_pefs[seq_len_key] = row["pef_path"].replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
                studio_model = row["model_path"].replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
        return sibling_studio_pefs if sibling_studio_pefs else None, studio_model   


    def _studio_only_rows(self) -> List[Dict]:
        studio_only = {k:v for k,v in self.studio_inventory.items() if k in self.studio_only_keys}
        rows = []
        for key, studio_row in studio_only.items():
            row = {}
            for field in InventoryComparer.studio_only_fields:
                if field in studio_row:
                    row[field] = studio_row[field]
            row["id"] = str(key)
            row["group_id"] = key.group_id
            rows.append(row) 
        
        return rows


    def _cloud_only_rows(self) -> List[Dict]:
        cloud_only = {k:v for k,v in self.cloud_inventory.items() if k in self.cloud_only_keys}
        rows = []
        for key, cloud_row in cloud_only.items():
            sibling_studio_pefs, studio_model = self._find_sibling_artifacts(key)
            row = {}
            for field in InventoryComparer.cloud_only_fields:
                if field in cloud_row:
                    row[field] = cloud_row[field]
            row["sibling_studio_pefs"] = sibling_studio_pefs
            row["studio_model"] = studio_model
            rows.append(row)
        
        return rows


    def _common_rows(self) -> List[Dict]:
        rows = []
        for key in self.common_keys:
            cloud_row, studio_row = self.cloud_inventory[key], self.studio_inventory[key]
            row = {}
            for field in InventoryComparer.common_fields:
                if field in studio_row:
                    row[field] = studio_row[field]
                if field in cloud_row:
                    row[field] = cloud_row[field]
                row["studio_model"] = replace_af_prefix(studio_row["model_path"])
                row["studio_pef"] = replace_af_prefix(studio_row["pef_path"])
                row["studio_batch_sizes"] = studio_row["batch_sizes"]
            sibling_studio_pefs, _ = self._find_sibling_artifacts(key)
            row["sibling_studio_pefs"] = sibling_studio_pefs 
            row_comparison_results = InventoryComparer._compare_rows(cloud_row, studio_row)
            row.update(row_comparison_results)
            rows.append(row)
        
        return rows


    def _onboard_to_studio_rows(self) -> List[Dict]:
        def _build_row(input_row: Dict, is_new_config: bool) -> Dict:
            row = {}
            for field in InventoryComparer.onboard_to_studio_fields:
                if field in input_row:
                    row[field] = input_row[field]
            row["onboard_cloud_models"] = row["studio_model"] is None
            row["is_new_config"] = is_new_config
            return row
        
        rows = []
        common_rows, cloud_only_rows = self._common_rows(), self._cloud_only_rows()
        for common_row in common_rows:
            # Only need to onboard rows in common where there are cloud-only BS PEFs or different PEFs for same BS
            if common_row["cloud_only_bs"] == [] and common_row["common_bs_different_pefs"] == []:
                continue
            rows.append(_build_row(common_row, False))
        for cloud_only_row in cloud_only_rows:
            rows.append(_build_row(cloud_only_row, True))
        
        return rows


if __name__ == "__main__":
    ic = InventoryComparer()
    ic.write()
