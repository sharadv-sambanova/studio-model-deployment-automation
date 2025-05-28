import yaml
import os
from schemas import InferenceDeployment, CloudConfig, InventoryKey
import csv
from utils import CLOUD_PROD_DEPLOYMENTS
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "output/cloud_inventory.csv"
GTM_OUTPUT_FILE = Path(__file__).parent / "output/cloud_inventory_gtm.csv"


def load_deployments():
    deployment_configs = [CLOUD_PROD_DEPLOYMENTS / f for f in os.listdir(CLOUD_PROD_DEPLOYMENTS)]
    deployments = {}
    inference_deployments = {}
    for config in deployment_configs:
        with open(config) as f:
            deployment = yaml.safe_load(f)
            deployments[config] = deployment
    for config, deployment in deployments.items():
        print(f"Processing {config}")
        d = InferenceDeployment(**deployment, deployment=config.stem)
        inference_deployments[config.stem] = d

    return inference_deployments


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


def write_inventory(configs: dict[InventoryKey, CloudConfig]):
    with open(OUTPUT_FILE, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CloudConfig.fieldnames)
        writer.writeheader()
        sorted_keys = sorted(configs.keys(), key=lambda x: str(x))
        for key in sorted_keys:
            row = configs[key].to_row()
            if row is not None:
                writer.writerow(row)


def write_inventory_gtm(configs: dict[InventoryKey, CloudConfig]):
    with open(GTM_OUTPUT_FILE, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CloudConfig.gtm_fieldnames)
        writer.writeheader()
        sorted_keys = sorted(configs.keys(), key=lambda x: str(x))
        rows = []
        for key in sorted_keys:
            rows += configs[key].to_gtm_rows()
        rows = sorted(rows, key=lambda x: (x["model_name"], x["spec_decoding"], x["max_seq_length"]))
        writer.writerows(rows)


if __name__ == "__main__":
    deployments = load_deployments()
    configs = get_cloud_configs(deployments)
    write_inventory(configs)
    write_inventory_gtm(configs)