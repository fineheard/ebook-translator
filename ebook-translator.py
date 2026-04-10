import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, JSONDecodeError

LM_STUDIO_URL = "http://localhost:1234"

def get_model_info():
    try:
        response = requests.get(f"{LM_STUDIO_URL}/api/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
    except ConnectionError:
        print("-" * 50)
        print("  [错误] 无法连接到 LM Studio")
        print("  请确保 LM Studio 正在运行且 API 服务已启用")
        print(f"  地址: {LM_STUDIO_URL}/api/v1/models")
        print("-" * 50)
        return None
    except Timeout:
        print("-" * 50)
        print("  [错误] 连接超时")
        print("  请检查网络连接或增加超时时间")
        print("-" * 50)
        return None
    except HTTPError as e:
        print("-" * 50)
        print(f"  [错误] HTTP 请求失败: {e.response.status_code}")
        print("  请检查 LM Studio 版本是否支持 v1 API")
        print("-" * 50)
        return None
    except JSONDecodeError:
        print("-" * 50)
        print("  [错误] 响应数据解析失败")
        print("  请检查 LM Studio 版本")
        print("-" * 50)
        return None
    
    models = data.get("models", [])
    loaded_models = [m for m in models if m.get("loaded_instances")]
    
    if not loaded_models:
        print("-" * 50)
        print("  [提示] 未找到已加载的模型")
        print("  请先在 LM Studio 中加载一个模型")
        print("-" * 50)
        return None
    
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

if __name__ == "__main__":
    get_model_info()
