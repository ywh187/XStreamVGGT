from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nha_fla.models.qwen2_nha.configuration_qwen2_nha import Qwen2NHAConfig
from nha_fla.models.qwen2_nha.modeling_qwen2_nha import Qwen2NHAForCausalLM, Qwen2NHAModel

AutoConfig.register(Qwen2NHAConfig.model_type, Qwen2NHAConfig)
AutoModel.register(Qwen2NHAConfig, Qwen2NHAModel)
AutoModelForCausalLM.register(Qwen2NHAConfig, Qwen2NHAForCausalLM)


__all__ = ['Qwen2NHAConfig', 'Qwen2NHAForCausalLM', 'Qwen2NHAModel']