import requests

LM_STUDIO_URL = "http://localhost:1234"

def get_model_info():
    try:
        response = requests.get(f"{LM_STUDIO_URL}/api/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        models = data.get("models", [])
        loaded_models = [m for m in models if m.get("loaded_instances")]
        
        if loaded_models:
            first_model = loaded_models[0]
            inst = first_model["loaded_instances"][0]
            
            caps = first_model.get('capabilities', {})
            
            print("=" * 50)
            print("       First Loaded Model")
            print("=" * 50)
            print(f"  display_name:       {first_model.get('display_name')}")
            print(f"  key:                {first_model.get('key')}")
            print(f"  type:               {first_model.get('type')}")
            print(f"  publisher:          {first_model.get('publisher')}")
            print(f"  architecture:       {first_model.get('architecture')}")
            print(f"  quantization:       {first_model['quantization']['name']} ({first_model['quantization']['bits_per_weight']} bits)")
            print(f"  format:             {first_model.get('format')}")
            print(f"  params_string:      {first_model.get('params_string')}")
            print(f"  size_bytes:         {first_model.get('size_bytes')} ({first_model.get('size_bytes') / (1024**3):.2f} GB)")
            print(f"  max_context_length: {first_model.get('max_context_length')}")
            print(f"  description:        {first_model.get('description')}")
            print("-" * 50)
            print("  loaded_instances:")
            print(f"    context_length:        {inst['config']['context_length']}")
            print(f"    eval_batch_size:       {inst['config']['eval_batch_size']}")
            print(f"    parallel:              {inst['config']['parallel']}")
            print(f"    flash_attention:       {inst['config']['flash_attention']}")
            print(f"    offload_kv_cache_to_gpu: {inst['config']['offload_kv_cache_to_gpu']}")
            print("-" * 50)
            print("  capabilities:")
            print(f"    vision:             {caps.get('vision')}")
            print(f"    trained_for_tool_use: {caps.get('trained_for_tool_use')}")
            print("=" * 50)
            
            return first_model
        else:
            print("No loaded models found.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    get_model_info()
