import yaml
import os
from schemas import InferenceDeployment, CloudConfig, InventoryKey
import csv
from utils import CLOUD_PROD_DEPLOYMENTS

OUTPUT_FILE = "output/cloud_inventory.csv"


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


def merge_deployments(deployments: dict[str, InferenceDeployment]):
    """Merge ModelConfigs across multiple deployments"""
    all_configs = None
    for deployment in deployments.values():
        # For the first deployment, just get all the configs
        if all_configs is None:
            all_configs = deployment.spec._model_configs
        else:
            # For subsequent deployments,
            for key, model_config in deployment.spec._model_configs.items():
                # If the config is one we already have, merge them together
                if key in all_configs:
                    all_configs[key].merge(model_config)
                # or add new ones
                else:
                    all_configs[key] = model_config

    return all_configs

def write_inventory(configs: dict[InventoryKey, CloudConfig]):
    with open(OUTPUT_FILE, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CloudConfig.fieldnames)
        writer.writeheader()
        sorted_keys = sorted(configs.keys(), key=lambda x: str(x))
        for key in sorted_keys:
            writer.writerow(configs[key].to_row())


if __name__ == "__main__":
    deployments = load_deployments()
    #configs = merge_deployments(deployments)
    configs = get_cloud_configs(deployments)
    write_inventory(configs)