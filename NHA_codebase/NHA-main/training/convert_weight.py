import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import argparse
import os

def pool_k_to_g(k_weight: torch.Tensor, target_s: int) -> torch.Tensor:
    source_s, d = k_weight.shape
    
    if source_s % target_s != 0:
        raise ValueError(f"The source dimension {source_s} must be divisible by the target dimension {target_s}.")
    
    factor = source_s // target_s
    k_weight_transposed = k_weight.transpose(0, 1)
    k_weight_unsqueezed = k_weight_transposed.unsqueeze(1)
    
    # (d, 1, source_s) -> (d, 1, target_s)
    pooled_transposed = F.avg_pool1d(k_weight_unsqueezed, kernel_size=factor, stride=factor)
    
    # (d, 1, target_s) -> (d, target_s) -> (target_s, d)
    result = pooled_transposed.squeeze(1).transpose(0, 1)
    return result

def convert_model_weights(input_path: str, output_path: str):
    """
    Loads a model, converts its weights, and saves the converted model.

    Args:
        input_path: Path to the input model directory.
        output_path: Path to the output directory to save the converted model.
    """
    print(f"Loading model from {input_path}...")
    # Load the model with float32 precision for the conversion process
    model = AutoModelForCausalLM.from_pretrained(input_path, torch_dtype=torch.float32, trust_remote_code=True)
    print("Model loaded successfully.")

    print("Starting weight conversion...")
    for name, param in model.named_parameters():
        if 'g_proj' in name and 'weight' in name:
            k_proj_name = name.replace('g_proj', 'k_proj')
            if k_proj_name in model.state_dict():
                k_param = model.get_parameter(k_proj_name)
                f_d = param.shape[0]
                print(f"Converting {k_proj_name} to {name}...")
                pooled_weight = pool_k_to_g(k_param.data, f_d)
                param.data.copy_(pooled_weight)
            else:
                print(f"Warning: Parameter {k_proj_name} not found, skipping conversion for {name}.")
    
    print("Weight conversion finished.")

    print("Converting model to bfloat16...")
    model.to(torch.bfloat16)

    print(f"Saving converted model to {output_path}...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    model.save_pretrained(
        output_path,
        max_shard_size="4GB",
    )
    print("Model saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Convert model weights.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory to save the converted model.")
    args = parser.parse_args()

    convert_model_weights(args.input_path, args.output_path)

if __name__ == "__main__":
    main()