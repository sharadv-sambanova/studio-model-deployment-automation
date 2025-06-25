import yaml
import os
from schemas import InferenceDeployment, CloudConfig, InventoryKey
import csv
from utils import CLOUD_PROD_DEPLOYMENTS, SN_IAC_PROD_CLUSTER_FILES
from pathlib import Path
from typing import Dict
from itertools import takewhile

OUTPUT_FILE = Path(__file__).parent / "output/cloud_inventory.csv"
GTM_OUTPUT_FILE = Path(__file__).parent / "output/cloud_inventory_gtm.csv"


def load_deployments(active_deployments):
    deployment_configs = [CLOUD_PROD_DEPLOYMENTS / f for f in os.listdir(CLOUD_PROD_DEPLOYMENTS)]
    deployments = {}
    inference_deployments = {}
    for config in deployment_configs:
        with open(config) as f:
            deployment = yaml.safe_load(f)
            # Only parse active deployments
            if deployment["metadata"]["name"] not in active_deployments:
                continue
            deployments[config] = deployment
    for config, deployment in deployments.items():
        print(f"Processing {config}")
        d = InferenceDeployment(**deployment, deployment=config.stem)
        inference_deployments[config.stem] = d

    return inference_deployments

def get_active_deployments():

    def read_coe_values_from_cluster_spec(cluster_spec):
        yaml_content = ""
        line = cluster_spec.readline()
        # Skip to the beginning of the coe-values.yaml definition
        while not "EOVAL" in line:
            line = cluster_spec.readline()
        # Read the first line of the yaml definition
        line = cluster_spec.readline()
        # Yaml is indented in the .tfvars file, need to unindent it
        indent = len(''.join(takewhile(str.isspace, line)))
        while not "EOVAL" in line:
            yaml_content += line[indent:]
            line = cluster_spec.readline()
        return yaml.safe_load(yaml_content)

    active_deployments = set()
    for cluster_file in SN_IAC_PROD_CLUSTER_FILES:
        with open(cluster_file) as f:
            coe_values = read_coe_values_from_cluster_spec(f)
        cluster_deployments = [d["name"] for d in coe_values['inferenceDeploymentSpecs']]
        active_deployments = active_deployments.union(cluster_deployments)
        
    return active_deployments


def get_cloud_configs(inference_deployments):
    cloud_configs = {}
    for deployment_name, inference_deployment in inference_deployments.items():
        spec = inference_deployment.spec
        for key, config in spec._cloud_configs.items():
            if key in cloud_configs:
                cloud_configs[key].merge(config)
            else:
                cloud_configs[key] = config
    return cloud_configs


def write_inventory(configs: Dict[InventoryKey, CloudConfig]):
    with open(OUTPUT_FILE, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CloudConfig.fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        sorted_keys = sorted(configs.keys(), key=lambda x: str(x))
        for key in sorted_keys:
            row = configs[key].to_row()
            if row is not None:
                writer.writerow(row)


def write_inventory_gtm(configs: Dict[InventoryKey, CloudConfig]):
    with open(GTM_OUTPUT_FILE, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CloudConfig.gtm_fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        sorted_keys = sorted(configs.keys(), key=lambda x: str(x))
        rows = []
        for key in sorted_keys:
            rows += configs[key].to_gtm_rows()
        rows = sorted(rows, key=lambda x: (x["model_name"], x["spec_decoding"], x["max_seq_length"]))
        writer.writerows(rows)


if __name__ == "__main__":
    active_deployments = get_active_deployments()
    deployments = load_deployments(active_deployments)
    configs = get_cloud_configs(deployments)
    write_inventory(configs)
    write_inventory_gtm(configs)
