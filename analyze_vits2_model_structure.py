import torch
from pathlib import Path
import json
from safetensors import safe_open
import argparse


def load_model(model_path):
    if model_path.suffix == '.safetensors':
        with safe_open(model_path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    elif model_path.suffix == '.pth':
        return torch.load(model_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported file format: {model_path.suffix}")


def analyze_model_structure(model_dict):
    sizes = set()
    important_shapes = {}
    for name, param in model_dict.items():
        if isinstance(param, torch.Tensor):
            if len(param.shape) > 0:
                sizes.add(param.shape[-1])
                if param.shape[-1] in [256, 512]:
                    important_shapes[name] = param.shape
    return sizes, important_shapes


def analyze_models(model_paths, config_path):
    config_path = Path(config_path)

    # configファイルを読み込む
    with open(config_path, 'r') as f:
        config = json.load(f)

    all_sizes = set()
    all_important_shapes = {}

    for model_path in model_paths:
        model_path = Path(model_path)
        print(f"\nAnalyzing {model_path.name}:")

        model_dict = load_model(model_path)
        if 'model' in model_dict:
            model_dict = model_dict['model']

        sizes, important_shapes = analyze_model_structure(model_dict)
        all_sizes.update(sizes)
        all_important_shapes.update(important_shapes)

        print(f"Unique sizes found: {sorted(sizes)}")
        print("Important shapes (256 or 512):")
        for name, shape in important_shapes.items():
            print(f"  {name}: shape = {shape}")

    print("\nOverall summary:")
    print(f"All unique sizes found across models: {sorted(all_sizes)}")

    print("\nImportant config information:")
    print(f"Model name: {config.get('model_name', 'Not specified')}")
    print(f"Version: {config.get('version', 'Not specified')}")
    print(f"Gin channels: {config['model'].get('gin_channels', 'Not specified')}")
    print(f"Hidden channels: {config['model'].get('hidden_channels', 'Not specified')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VITS2 model files")
    parser.add_argument("config_path", type=str, help="Path to the config.json file")
    parser.add_argument("model_paths", type=str, nargs='+', help="Paths to the model files (.pth or .safetensors)")

    args = parser.parse_args()

    analyze_models(args.model_paths, args.config_path)
