from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nha_fla.models.llama3_nha.configuration_llama3_nha import LlamaNHAConfig
from nha_fla.models.llama3_nha.modeling_llama3_nha import LlamaNHAForCausalLM, LlamaNHAModel

AutoConfig.register(LlamaNHAConfig.model_type, LlamaNHAConfig)
AutoModel.register(LlamaNHAConfig, LlamaNHAModel)
AutoModelForCausalLM.register(LlamaNHAConfig, LlamaNHAForCausalLM)


__all__ = ['LlamaNHAConfig', 'LlamaNHAForCausalLM', 'LlamaNHAModel']