from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nha_fla.models.qwen3_moe_nha.configuration_qwen3_moe_nha import Qwen3MoeNHAConfig
from nha_fla.models.qwen3_moe_nha.modeling_qwen3_moe_nha import Qwen3MoeNHAForCausalLM, Qwen3MoeNHAModel

AutoConfig.register(Qwen3MoeNHAConfig.model_type, Qwen3MoeNHAConfig)
AutoModel.register(Qwen3MoeNHAConfig, Qwen3MoeNHAModel)
AutoModelForCausalLM.register(Qwen3MoeNHAConfig, Qwen3MoeNHAForCausalLM)

__all__ = ['Qwen3MoeNHAConfig', 'Qwen3MoeNHAForCausalLM', 'Qwen3MoeNHAModel']