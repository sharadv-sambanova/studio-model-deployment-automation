import re
import yaml
from pathlib import Path
from typing import Union, Dict, List
import csv

DAAS_RELEASE_ROOT = Path(__file__).parent.parent.parent / "daas-release"
FAST_COE_ROOT = Path(__file__).parent.parent.parent / "fast-coe"
SN_IAC_ROOT = Path(__file__).parent.parent.parent / "sn_iac"

SN_IAC_PROD = SN_IAC_ROOT / "environments" / "production" / "terraform" / "modules" / "sn_vcluster_tenant_v2" / "tfvars"
SN_IAC_PROD_CLUSTER_FILES = [SN_IAC_PROD / f"{cluster_name}.tfvars" for cluster_name in 
    [
        "fast-snova-ai-jp-prod-2",
        "fast-snova-ai-prod-0",
        "fast-snova-ai-prod-1",
    ]
]

STUDIO_INVENTORY_PATH= DAAS_RELEASE_ROOT / "inventory/inventory_output/prod/models_and_pefs_gtm.csv"
CLOUD_MODELS_YAML = FAST_COE_ROOT / "helm/values.yaml"
CLOUD_PROD_DEPLOYMENTS = FAST_COE_ROOT / "helm/inference-deployments/prod"

MODEL_MAPPINGS_FILE = Path(__file__).parent / "model_arch_mappings.yaml"
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

def convert_seq_len(seq_len, target_type=str) -> Union[str, int]:
    """Convert a seq_len value to target_type"""
    supported_types = [int, str]
    if target_type not in supported_types:
        raise ValueError("Only 'str' and 'int' are supported for target_type")
    if not any([isinstance(seq_len, x) for x in supported_types]):
        raise ValueError("Only 'str' and 'int' are supported for seq_len")
    
    # No conversion needed
    if isinstance(seq_len, target_type):
        return seq_len
    # Lookup in above map to convert str -> int
    if isinstance(seq_len, int):
        return MAX_SEQ_LEN_MAP[seq_len]
    # Lookup in reverse of above map to convert int -> str
    elif isinstance(seq_len, str):
        reverse_map = {v:k for k,v in MAX_SEQ_LEN_MAP.items()}
        return reverse_map[seq_len]


def lookup_seq_len(expert_name: str) -> int:
    """Lookup the seq len for an expert in the cloud helm chart"""
    for model_name, model in CLOUD_MODELS.items():
        try:
            if model_name == expert_name or expert_name in model.get("aliases", {}):
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
        except KeyError as e:
            return 4096
    return 4096


def get_expert_seq_len(expert_name: str) -> int:
    """Return the expert's sequence length as an int, based on the expert_name or helm/values.yaml"""
    # match '-<digits>k' at the end of the expert name
    seq_len = None
    match = re.search(r'-(\d+)k$', expert_name)

    if expert_name == "Llama-4-Maverick-17B-128E-Instruct-Text": # Hack to handle just this expert
        seq_len = "8k"
    elif match:
        # discard the '-'
        seq_len = match.group()[1:]
    # If no seq len specified in expert name, lookup in values.yaml
    else:
        seq_len = lookup_seq_len(expert_name)

    return convert_seq_len(seq_len, int)


def get_pef_jira(pef_path: str) -> str:
    """Extract the PEF Jira from the pef_path, or return None if no match was found"""
    pef_path = pef_path.lower()
    pat = r'(?i)pef[-_]+\d+'
    match = re.search(pat, pef_path) # e.g. pef__-1234
    if match is None:
        return None
    jira = match.group().upper() # e.g. PEF__-1234
    jira = jira.replace('-','').replace('_','') # e.g. PEF1234
    jira = jira[:3] + "-" + jira[3:] # e.g. PEF-1234
    return jira

def normalize_expert_name(expert_name) -> str:
    """Remove the trailing sequence length marker from an expert name"""
    if expert_name == "Llama-4-Maverick-17B-128E-Instruct-Text": # Hack to handle just this expert
        return "Llama-4-Maverick-17B-128E-Instruct"
    if re.search(r'-(\d+)k$', expert_name):
        return "-".join(expert_name.split("-")[:-1])
    else:
        return expert_name


def get_mapping(expert_name) -> Dict[str, str]:
    """Lookup the expert's normalized name in the MODEL_MAPPINGS_FILE"""

    class UnknownExpertError(Exception):
        pass

    expert_name_norm = normalize_expert_name(expert_name)
    expert_mapping = MODEL_MAPPINGS.get(expert_name_norm, None)
    if expert_mapping is None:
        raise UnknownExpertError(f"Could not find normalized expert name {expert_name_norm} in mappings file {MODEL_MAPPINGS_FILE}")
    return expert_mapping


def get_app_name(expert_name) -> str:
    """Get the app name from the expert's MODEL_MAPPINGS_FILE entry"""
    expert_mapping = get_mapping(expert_name)
    app_name = expert_mapping["app_name"]
    if app_name is None:
        return normalize_expert_name(expert_name)
    return app_name


def get_parameter_count(expert_name) -> str:
    """Get the model_parameter_count from the expert's MODEL_MAPPINGS_FILE entry"""
    expert_mapping = get_mapping(expert_name)
    return str(expert_mapping["model_parameter_count"])


def replace_af_prefix(inp: str) -> str:
    """Replace prefix variables from artifactory paths with their corresponding dev repo"""
    inp = inp.replace("{{ARTIFACTS_REPO}}", "sw-generic-daas-artifacts-dev")
    return inp

def read_csv(csv_file) -> List[Dict]:
    """Return all the rows from a csv"""
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows
