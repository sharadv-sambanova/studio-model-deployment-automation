
from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Dict, Optional, Union
from utils import get_expert_seq_len, get_app_name, get_parameter_count, normalize_expert_name, get_pef_jira, convert_seq_len
from dataclasses import dataclass, fields
import json

######## Pydantic classes for cloud deployment yamls ############

class Metadata(BaseModel):
    name: str


class PEFData(BaseModel):
    source: str


class CheckpointData(BaseModel):
    source: str


class Expert(BaseModel):
    batch_size: int
    pef: str
    copy_pef: Optional[str] = None
    checkpoint: str
    num_tokens_at_a_time: Optional[int] = Field(default=None)
    ckpt_sharing: bool
    ckpt_sharing_uuid: Optional[str] = Field(default=None)
    private: Optional[bool] = Field(default=None)


class SpeculativeDecodingConfig(BaseModel):
    batch_size: int
    k: int
    draft_model: str
    target_model: str


class Spec(BaseModel):
    environmentSecretNames: List[str]
    pefs: Dict[str, PEFData]
    checkpoints: Dict[str, CheckpointData]
    experts: Dict[str, List[Expert]]
    speculative_decoding: Optional[List[SpeculativeDecodingConfig]] = Field(default_factory = list)

    # The model configs defined by this inference deployment spec
    # Private attribute, not used by pydantic
    _cloud_configs: dict = PrivateAttr(default_factory = dict)

    def model_post_init(self, __context):
        if not hasattr(self, '_model_configs'):
            self._model_configs = {}
        self._add_cloud_configs()

    def _add_cloud_configs(self) -> Dict["InventoryKey", "CloudConfig"]:
        for expert_name, experts in self.experts.items():
            new_config = CloudConfig(expert_name, experts, self)
            self._cloud_configs[new_config.key] = new_config

    def set_deployment(self, deployment: str):
        for _, cloud_config in self._cloud_configs.items():
            cloud_config.add_deployment(deployment)


class InferenceDeployment(BaseModel):
    apiVersion: str
    kind: str
    metadata: Metadata
    spec: Spec

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "deployment" in kwargs:
            self.spec.set_deployment(kwargs["deployment"])


######## Custom classes ############


@dataclass(frozen=True)
class InventoryKey:
    """
        Class representing a key in the inventory
        Each row in the inventory represents a unique combination of the below parameters

        Instances of this class must be instantiated with from_input()
    """
    app_name: str
    param_count: str
    sd: bool
    max_seq_length: int

    fields_aliases = {
        "app_name": ["app_name", "model_app_name"],
        "param_count": ["param_count", "model_parameter_count"],
        "sd": ["sd", "spec_decoding", "speculative_decoding"],
        "max_seq_length": ["max_seq_length", "max_seq_len"]
    }

    @classmethod
    def lookup_field(cls, fieldname: str, obj):
        """Get the value of fieldname from obj. obj may be a dict or an arbitrary object"""
        if isinstance(obj, dict):
            for alias in InventoryKey.fields_aliases[fieldname]:
                if alias in obj:
                    return obj[alias]
            raise KeyError
        else:
            for alias in InventoryKey.fields_aliases[fieldname]:
                if hasattr(obj, alias):
                    return getattr(obj, alias)
            raise AttributeError

    @classmethod
    def from_input(cls, obj: Union[Dict, object]) -> "InventoryKey":
        """
            Instantiates an InventoryKey based on a dict or an arbitrary object
            For dict-based instantiation the dict must have a key for each field in field(InventoryKey)
            For object-based instantiation the dict must have an attribute for each field in field(InventoryKey)
        """

        def to_bool(val):
            """Cast"""
            if not isinstance(val, str):
                return bool(val)
            val_lower = val.lower()
            if val_lower == "true":
                return True
            elif val_lower == "false":
                return False
            else:
                raise ValueError(f"Invalid str for bool conversion: {val}")

        class InvalidDictForInventoryKey(Exception):
            pass

        class InvalidObjectForInventoryKey(Exception):
            pass

        init_kwargs = {}
        field_names = {f.name for f in fields(cls)}
        try:
            for field in fields(cls):
                # Get the value of the field from obj, cast it to the right type, store it in init_kwargs
                if field.type == bool:
                    val = to_bool(InventoryKey.lookup_field(field.name, obj))
                else:
                    val = field.type(InventoryKey.lookup_field(field.name, obj))
                init_kwargs[field.name] = val
        except KeyError:
            raise InvalidDictForInventoryKey(
                f"Expected keys {field_names} in dict used for InventoryKey initialization, got {sorted(list(obj.keys()))}"
            )
        except AttributeError:
            raise InvalidObjectForInventoryKey(
                f"Expected attributes {field_names} in object used for InventoryKey initialization, got {sorted(list(dir(obj)))}"
            )

        return cls(**init_kwargs)

    def __str__(self):
        """Return this InventoryKey's fields joined together with '-' and spaces replaced with '_'"""
        s = "-".join(tuple(str(getattr(self, x.name)) for x in fields(InventoryKey)))
        s = s.replace(" ", "_") # App name has spaces
        s = s.replace(".", "d") # App name might have "."
        return s

    @property
    def group_id(self):
        """Return the group ID of this inventory key, which consists of all fields besides max_seq_length"""
        return "-".join(str(self).split("-")[:-1])


    def is_sibling(self, other_key: "InventoryKey"):
        """Check if this InventoryKey is a sibling of other_key. Sibling means Studio would load the artifacts for these keys together"""
        return  self.group_id == other_key.group_id


class PEF():
    """
        Represents a single Cloud PEF (one batch size)
    """
    def __init__(self, expert: Expert, pefdata: PEFData, is_sd: bool):
        self.name: str = expert.pef
        self.batch_size: int = expert.batch_size
        self.path: str = pefdata.source
        self.jira: str = get_pef_jira(self.path)
        self.copy_pef: str = expert.copy_pef
        self.sd: bool = is_sd

    def __eq__(self, other_pef: "PEF"):
        return self.path == other_pef.path

    def as_dict(self) -> dict[str, str]:
        return {
            self.batch_size: 
            {
                "pef_path": self.path,
                "copy_pef": self.copy_pef,
                "jira": self.jira,
            }
        }

    def __hash__(self):
        return hash(self.path)


class CloudConfig():
    """
        Class representing a row in the inventory
    """
    def __init__(self, name: str, experts: List[Expert], spec: Spec):
        self.name: str = name
        self.expert_to_checkpoint = {normalize_expert_name(self.name): self.get_checkpoint_path(experts, spec)}
        self.max_seq_length: int = get_expert_seq_len(self.name)
        self.param_count: str = get_parameter_count(self.name)
        self.app_name: str = get_app_name(self.name)
        self.deployments: str = set()
        self.sd, self.draft_experts = self.process_sd(spec)
        self.pefs: Dict[str, PEF] = self.build_pefs(experts, spec)
        self.key = InventoryKey.from_input(self)


    @property
    def batch_sizes(self) -> int:
        return sorted({p.batch_size for p in self.pefs.values()})


    @property
    def id(self) -> str:
        return str(self.key)
    
    @property
    def group_id(self) -> str:
        return self.key.group_id

    def add_deployment(self, deployment: str):
        self.deployments.add(deployment)


    def __str__(self):
        return f"app_name: {self.app_name} | param_count: {self.param_count} | max_seq_length: {self.max_seq_length} | sd: {self.sd}"


    def get_checkpoint_path(self, experts: List[Expert], spec: Spec) -> str:
        """Return the GCS path of the checkpoint this expert references. Each expert should have 1 unique checkpoint path."""

        class NonUniqueCheckpointError(Exception):
            pass

        unique_checkpoints = {e.checkpoint for e in experts}
        if len(unique_checkpoints) != 1:
            raise NonUniqueCheckpointError(f"Expert {self.name} does not have exactly 1 unique checkpoint")
        return spec.checkpoints[unique_checkpoints.pop()].source


    def process_sd(self, spec: Spec) -> tuple[bool, set]:
        """Return whether this CloudConfig is a target model, and corresponding draft models if so"""
        sd, draft_experts = False, set()
        for sd_config in spec.speculative_decoding:
            if sd_config.target_model == self.name:
                sd = True
                draft_experts.add(normalize_expert_name(sd_config.draft_model))
        
        return sd, draft_experts


    def build_pefs(self, experts: list[Expert], spec: Spec) -> dict[str, PEF]:
        """Process the list of experts to create PEF objects for this CloudConfig"""

        class NonUniquePEFsError(Exception):
            pass

        pefs = {}
        for expert in experts:
            if expert.pef in pefs:
                raise NonUniquePEFsError(f"Expert {self.name} references pef {expert.pef} multiple times")
            new_pef = PEF(expert,spec.pefs[expert.pef], self.sd)
            pefs[expert.pef] = new_pef
        return pefs


    fieldnames = ["id", "group_id", "model_app_name", "experts", "deployments", "param_count", "max_seq_length", "max_seq_length_cloud", "spec_decoding", "batch_sizes", "cloud_pefs_json", "draft_experts"]
    def to_row(self) -> dict:
        """Return a dict representing this CloudConfig to be used for writing to a csv with DictWriter"""

        cloud_pefs_json = {}
        for pef in self.pefs.values():
            cloud_pefs_json.update(pef.as_dict())
        return {
            "id": self.id,
            "group_id": self.group_id,
            "model_app_name": self.app_name,
            "experts": sorted(list(self.expert_to_checkpoint.keys())),
            "deployments": self.deployments,
            "param_count": self.param_count, 
            "max_seq_length": convert_seq_len(self.max_seq_length, int), 
            "max_seq_length_cloud": convert_seq_len(self.max_seq_length, str),
            "spec_decoding": self.sd, 
            "batch_sizes": self.batch_sizes, 
            "cloud_pefs_json": json.dumps(cloud_pefs_json),
            "draft_experts": sorted(list(self.draft_experts))
        }


    def merge(self, other_config: "CloudConfig"):
        """Update this CloudConfig with the artifacts from the other_config"""
        self.pefs.update(other_config.pefs)
        self.expert_to_checkpoint.update(other_config.expert_to_checkpoint)
        self.draft_experts = self.draft_experts.union(other_config.draft_experts)
        self.deployments = self.deployments.union(other_config.deployments)
