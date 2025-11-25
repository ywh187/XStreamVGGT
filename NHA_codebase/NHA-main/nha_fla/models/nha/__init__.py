# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nha_fla.models.nha.configuration_nha import NHAConfig
from nha_fla.models.nha.modeling_nha import NHAForCausalLM, NHAModel

AutoConfig.register(NHAConfig.model_type, NHAConfig)
AutoModel.register(NHAConfig, NHAModel)
AutoModelForCausalLM.register(NHAConfig, NHAForCausalLM)


__all__ = ['NHAConfig', 'NHAForCausalLM', 'NHAModel']
