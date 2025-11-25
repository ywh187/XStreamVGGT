from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nha_fla.models.qwen3_nha.configuration_qwen3_nha import Qwen3NHAConfig
from nha_fla.models.qwen3_nha.modeling_qwen3_nha import Qwen3NHAForCausalLM, Qwen3NHAModel

AutoConfig.register(Qwen3NHAConfig.model_type, Qwen3NHAConfig)
AutoModel.register(Qwen3NHAConfig, Qwen3NHAModel)
AutoModelForCausalLM.register(Qwen3NHAConfig, Qwen3NHAForCausalLM)

__all__ = ['Qwen3NHAConfig', 'Qwen3NHAForCausalLM', 'Qwen3NHAModel']