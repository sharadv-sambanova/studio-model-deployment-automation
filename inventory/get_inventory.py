import yaml
import os
from schemas import InferenceDeployment, ModelConfig
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

def write_inventory(configs: dict[str, ModelConfig]):
    with open(OUTPUT_FILE, "w") as f:
        writer = csv.DictWriter(f, fieldnames=ModelConfig.fieldnames)
        writer.writeheader()
        for config in configs.values():
            writer.writerow(config.to_row())


if __name__ == "__main__":
    deployments = load_deployments()
    configs = merge_deployments(deployments)
    write_inventory(configs)