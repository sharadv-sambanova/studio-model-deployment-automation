
from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Dict, Optional
from utils import lookup_seq_len, get_app_name, get_parameter_count, normalize_expert_name, get_pef_jira, MAX_SEQ_LEN_MAP
import json


class Metadata(BaseModel):
    name: str


class PEFData(BaseModel):
    source: str

    # Private attributes, not used by pydantic
    _batch_size: int = PrivateAttr(default=None)
    _jira: str = PrivateAttr(default=None)
    _copy_pef: "PEFData" = PrivateAttr(default=None)
    _ckpt_sharing_uuid: str = PrivateAttr(default=None)

    def model_post_init(self, __context):
        self._jira = get_pef_jira(self.source)


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

    # Private attributes, not used by pydantic
    _is_processed: bool = PrivateAttr(default=False)
    _sd: bool = PrivateAttr(default=False)
    _max_seq_len: int = PrivateAttr(default=None)
    _app_name: str = PrivateAttr(default=None)
    _model_parameter_count: str = PrivateAttr(default=None)
    _pef_data: PEFData = PrivateAttr(default=None)
    _checkpoint_data: CheckpointData = PrivateAttr(default=None)
    _deployment: str = PrivateAttr(default=None)

    # Model name -> path
    _draft_models: Dict[str, str] = PrivateAttr(default_factory=dict)


class SpeculativeDecodingConfig(BaseModel):
    batch_size: int
    k: int
    draft_model: str
    target_model: str


class ModelConfig():
    fieldnames = ["id", "model_app_name", "experts", "deployments", "param_count", "max_seq_len", "max_seq_len_cloud", "spec_decoding", "batch_sizes", "cloud_pefs_json", "pefs", "copy_pefs"]
    
    def __init__(self, expert: Expert, expert_name):
        if not expert._is_processed:
            raise ValueError("ModelConfig cannot be instantiated from unprocessed Expert")
        self.sd: bool = expert._sd
        self.max_seq_len: int = expert._max_seq_len
        self.app_name: str = expert._app_name
        self.model_parameter_count: str = expert._model_parameter_count

        self.experts = {normalize_expert_name(expert_name)}
        # Below dicts store artifact name -> data / path
        self.pefs: Dict[str, PEFData] = {expert.pef: expert._pef_data}
        self.checkpoints: Dict[str, str] = {expert.checkpoint: expert._checkpoint_data.source}
        self.draft_models: Dict[str, str] = expert._draft_models

        self.deployments = set()

    def to_row(self):
        """Return a dict representing this ModelConfig to be used for writing to a csv with DictWriter"""

        def get_formatted_pefs():
            """Return a dict representing this ModelConfig's PEFs"""
            pef_info = {}
            for pefdata in self.pefs.values():
                pef_info[pefdata._batch_size] = {
                    "pef_path": pefdata.source,
                    "copy_pef": pefdata._copy_pef.source if pefdata._copy_pef is not None else None,
                    "jira": pefdata._jira
                }
            return pef_info

        return {
            "id": self.id,
            "model_app_name": self.app_name,
            "experts": sorted(list(self.experts)),
            "deployments": self.deployments,
            "param_count": self.model_parameter_count, 
            "max_seq_len": self.max_seq_len, 
            "max_seq_len_cloud": MAX_SEQ_LEN_MAP[self.max_seq_len], 
            "spec_decoding": self.sd, 
            "batch_sizes": self.batch_sizes, 
            "cloud_pefs_json": get_formatted_pefs(),
            "pefs": [p.source for p in self.pefs.values()],
            "copy_pefs": [copy_p.source for copy_p in [p._copy_pef for p in self.pefs.values() if p._copy_pef is not None]],
        }


    def add_expert(self, expert: Expert, expert_name: str):
        if not expert._is_processed:
            raise ValueError(f"Unprocessed Expert cannot be added to a ModelConfig {str(self)}")
        self.pefs[expert.pef] = expert._pef_data
        self.checkpoints[expert.checkpoint] = expert._checkpoint_data.source
        self.draft_models.update(expert._draft_models)
        self.experts.add(normalize_expert_name(expert_name))


    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.key)

    @property
    def batch_sizes(self):
        return sorted(list({p._batch_size for p in self.pefs.values()}))

    @property
    def key(self):
        return (self.app_name, self.model_parameter_count, self.max_seq_len, self.sd)

    @property
    def id(self):
        id = "_".join((self.app_name, self.model_parameter_count, str(self.sd), self.max_seq_len))
        id = "_".join(id.split())
        id = id.replace(".", "d")
        return id

    def __repr__(self):
        return f"app_name: {self.app_name} | parameter_count: {self.model_parameter_count} | max_seq_len: {self.max_seq_len} | sd: {self.sd}"

    def merge(self, other_config: "ModelConfig"):
        """Update this ModelConfig with the artifacts from the other_config"""
        self.pefs.update(other_config.pefs)
        self.checkpoints.update(other_config.checkpoints)
        self.draft_models.update(other_config.draft_models)
        self.experts = self.experts.union(other_config.experts)
        self.deployments = self.deployments.union(other_config.deployments)
        
    def _validate_checkpoint_sharing(self):
        """Check if all the PEFs for this ModelConfig share the same checkpoint sharing UUID"""
        _ckpt_sharing_uuids = {p._ckpt_sharing_uuid for p in self.pefs.values() if p._ckpt_sharing_uuid is not None}
        assert len(_ckpt_sharing_uuids) < 2, f"Got multiple ckpt sharing UUIDs for PEFs in the same config \
        \nconfig: {str(self)}\npefs: {self.pefs}\nckpt sharing UUIDs: {_ckpt_sharing_uuids}"


class Spec(BaseModel):
    environmentSecretNames: List[str]
    pefs: Dict[str, PEFData]
    checkpoints: Dict[str, CheckpointData]
    experts: Dict[str, List[Expert]]
    speculative_decoding: Optional[List[SpeculativeDecodingConfig]] = Field(default_factory = list)
    _deployment: str = PrivateAttr(default=None)

    # The model configs defined by this inference deployment spec
    # Private attribute, not used by pydantic
    _model_configs: PrivateAttr(default_factory = dict)

    def model_post_init(self, __context):
        if not hasattr(self, '_model_configs'):
            self._model_configs = {}

        for expert_name, experts in self.experts.items():
            sd = False
            for sd_config in self.speculative_decoding:
                if sd_config.target_model == expert_name:
                    sd = True
                    sd_draft_checkpoints = self._get_checkpoints_for_expert(sd_config.draft_model)
                    expert._draft_models.update(sd_draft_checkpoints)
            max_seq_len = lookup_seq_len(expert_name)

            for expert in experts:
                expert._sd = sd
                expert._max_seq_len = max_seq_len
                self._process_expert_pefs(expert)

                expert._pef_data = self.pefs[expert.pef]
                expert._checkpoint_data = self.checkpoints[expert.checkpoint]

                expert._app_name = get_app_name(expert_name)
                expert._model_parameter_count = get_parameter_count(expert_name)
                expert._is_processed = True

                self._add_model_config(expert, expert_name)

    def set_deployment(self, deployment: str):
        for _, model_config in self._model_configs.items():
            model_config.deployments.add(deployment)

    def _process_expert_pefs(self, expert: Expert):
        """Update the PEFData corresponding to an Expert's pef and copy_pef"""
        def _update_pefdata(pefdata_to_update):
            """Add PEF-related properties from an Expert to a PEFData"""
            pefdata_to_update._batch_size = expert.batch_size
            pefdata_to_update._ckpt_sharing_uuid = expert.ckpt_sharing_uuid

        _update_pefdata(self.pefs[expert.pef])
        if expert.copy_pef is not None:
            copy_pefdata = self.pefs[expert.copy_pef]
            _update_pefdata(copy_pefdata)
            # Link the copy_pef to to pef
            self.pefs[expert.pef]._copy_pef = copy_pefdata

    def _get_checkpoints_for_expert(self, expert_name):
        checkpoint_names = [expert.checkpoint for expert in self.experts[expert_name]]
        name_to_path = {}
        for name in checkpoint_names:
            name_to_path[name] = self.checkpoints[name].source
        return name_to_path
    
    def _add_model_config(self, expert: Expert, expert_name: str):
        config = ModelConfig(expert, expert_name)
        key = config.key
        # If an expert matches an existing model config, add this expert's checkpoint / PEF to that config
        if key in self._model_configs:
            config = self._model_configs[key]
            config.add_expert(expert, expert_name)
        
        self._model_configs[key] = config
    

class InferenceDeployment(BaseModel):
    apiVersion: str
    kind: str
    metadata: Metadata
    spec: Spec

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "deployment" in kwargs:
            self.spec.set_deployment(kwargs["deployment"])
