import re
import yaml
from pathlib import Path

DAAS_RELEASE_ROOT = Path("/Users/sharadv/code/daas-release/")
FAST_COE_ROOT = Path("/Users/sharadv/code/fast-coe")

STUDIO_INVENTORY_PATH= DAAS_RELEASE_ROOT / "inventory/inventory_output/prod/models_and_pefs_gtm.csv"
CLOUD_MODELS_YAML = FAST_COE_ROOT / "helm/values.yaml"
CLOUD_PROD_DEPLOYMENTS = FAST_COE_ROOT / "helm/inference-deployments/prod"

MODEL_MAPPINGS_FILE = Path("/Users/sharadv/code/studio-model-deployment-automation/inventory/model_mappings.yaml")
with open(MODEL_MAPPINGS_FILE) as f:
    MODEL_MAPPINGS=yaml.safe_load(f)
with open(CLOUD_MODELS_YAML) as f:
    CLOUD_MODELS = yaml.safe_load(f)["models"]

MAX_SEQ_LEN_MAP = {
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
    65536: "64k",
    131072: "128k"
}


class UnknownExpertError(Exception):
    pass

def lookup_seq_len(expert_name: str):
    """Lookup the seq len for an expert in the cloud helm chart"""
    for model in CLOUD_MODELS:
        if expert_name in model["aliases"]:
            # CLOUD_MODELS format:
            #   <model name>:
                #   aliases:
                #       - <expert_name>:
                #   gatewayTokenizer:
                #       backendModels:
                #           - name: <expert_name>
                #           - maxSequenceLength: <max_seq_len>
            backend_model = [m for m in model["gatewayTokenizer"]["backendModels"] if m["name"] == expert_name][0]
            return backend_model["maxSequenceLength"]


# def get_expert_seq_len(expert_name: str):
#     # match '-<digits>k' at the end of the expert name
#     match = re.search(r'-(\d+)k$', expert_name)
#     if match:
#         # discard the '-'
#         return match.group()[1:]
#     # If no seq len specified in expert name, default to 4k
#     return "4k"

def get_pef_jira(pef_path: str):
    pef_path = pef_path.lower()
    pat = r'(?i)pef[-_]+\d+'
    match = re.search(pat, pef_path)
    if match is None:
        return None
    jira = match.group().upper()
    jira = jira.replace('-','').replace('_','')
    jira = jira[:3] + "-" + jira[3:]
    return jira

def normalize_expert_name(expert_name):
    if re.search(r'-(\d+)k$', expert_name):
        return "-".join(expert_name.split("-")[:-1])
    else:
        return expert_name


def get_mapping(expert_name):
    expert_name_norm = normalize_expert_name(expert_name)
    expert_mapping = MODEL_MAPPINGS.get(expert_name_norm, None)
    if expert_mapping is None:
        raise UnknownExpertError(f"Could not find normalized expert name {expert_name_norm} in mappings file {MODEL_MAPPINGS_FILE}")
    return expert_mapping


def get_app_name(expert_name):
    expert_mapping = get_mapping(expert_name)
    app_name = expert_mapping["app_name"]
    if app_name is None:
        return normalize_expert_name(expert_name)
    return app_name


def get_parameter_count(expert_name):
    expert_mapping = get_mapping(expert_name)
    return str(expert_mapping["model_parameter_count"])

if __name__ == "__main__":
    pef_path = 'gs://acp-coe-models-checkpoints-prod-0/version/deepseek-0.0.4/pefs-checkpoints/pefs/PEF_1663_multi_prefill/FP8_trial_Deepseek_V3_TP16_ss1024_16384_24576_32768__1024_4096_8192_12288_16384_token_gen_seq_length2_61_encs_flags__CoE_ckpt_sharing_BS1_SS1024_16384_24576_32768.pef'
    get_pef_jira(pef_path)