import requests
import os
import time
import argparse
import zipfile
import tempfile
import re
import glob
import uuid
import json
import logging
import sys
import shutil
import concurrent.futures
import math
from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
import warnings
from datetime import datetime

# ---------- 配置 ----------
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"
ACTUAL_MODEL_NAME = None
MODEL_CONTEXT_LENGTH = None
MAX_TOKENS = None          # 动态计算
CHUNK_SIZE = None          # 动态计算
MAX_WHOLE_ATTEMPT_LIMIT = 12000
DEFAULT_MAX_WORKERS = 2    # 默认并发线程数

CONNECT_TIMEOUT = 60
READ_TIMEOUT = 600          # 增加到 10 分钟，避免超时

# 批处理开关
BATCH_ENABLED = True

API_CALL_COUNT = 0

# 全局变量存储模型元数据
MODEL_PUBLISHER = None
MODEL_DISPLAY_NAME = None
MODEL_QUANT_NAME = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------- 翻译专家配置 ----------
PROFILES = [
    {
        "name": "技术翻译",
        "system_prompt": "你是专业技术书籍翻译专家。将以下英文翻译成准确、流畅、保留技术术语的中文，并确保全文术语翻译一致。不要使用脚注、括号注释或名词解释。只输出翻译结果，不要加任何解释或多余文字。",
        "temperature": 0.3
    },
    {
        "name": "文学翻译",
        "system_prompt": "你是专业文学翻译专家，擅长翻译英文小说。将以下英文翻译成优美、自然、保留原著情感和风格的中文。注意：\n- 保留人物的语气、性格和情感色彩。\n- 对话翻译需口语化、自然，符合人物身份。\n- 文化元素可适当归化，但关键意象和隐喻尽量保留。\n- 保持全文风格一致（如浪漫、悬疑、情色等）。\n- 对于人物姓名，保持翻译一致性，采用通用译名或根据上下文确定。\n- 对于情色内容，保留原文的直白程度，不刻意删减或美化。\n- 不要添加任何脚注、名词解释、括号注释或额外说明。\n直接输出翻译结果，不要有任何思考过程或解释。",
        "temperature": 0.4
    },
    {
        "name": "社科非虚构翻译",
        "system_prompt": """你是专业社科与科普书籍翻译专家，擅长翻译网络科学、复杂系统、行为经济学、社会心理学、决策科学等非虚构著作。将以下英文翻译成准确、严谨、流畅且自然易读的中文。必须严格遵守以下规则：
- **术语一致性**：所有专业术语必须全书保持100%一致。首次出现时在括号内标注英文原词，后续重复出现时直接使用已确定的译名，不再重复标注。
- **数据与细节**：严格保留原文中的所有数据、实验描述、统计结果、引用和案例细节，不增不减、不改动数字、不添加解释。
- **语气风格**：保持原著理性、科学、略带幽默的语气（不要变成冷冰冰的教科书，也不要过度文艺）。
- **逻辑层次**：句子逻辑清晰、层次分明，适合中国读者阅读，但绝不牺牲原意或简化论证。
- **专有名词**：全书人称、书名、专有名词保持一致。
- 不要添加任何脚注、名词解释、括号注释或额外说明。
直接输出翻译结果，不要有任何思考过程、解释或额外文字。""",
        "temperature": 0.38
    }
]

IGNORE_TAGS = {'pre', 'code', 'script', 'style', 'table', 'svg', 'math'}

# ---------- 获取模型详细信息（优先使用运行时 context_length） ----------
def get_model_details():
    """
    从 LM Studio 获取当前加载模型的详细信息。
    优先访问 http://localhost:1234/api/v1/models，解析：
    - 运行时上下文长度：loaded_instances[0].config.context_length（用户实际设置的值）
    - publisher, display_name, quantization.name
    返回 (model_id, publisher, display_name, quant_name, context_length) 元组。
    """
    api_url = "http://localhost:1234/api/v1/models"
    try:
        resp = requests.get(api_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            logger.debug(f"/api/v1/models 响应片段: {json.dumps(data, ensure_ascii=False)[:200]}...")
            models_list = data.get('models', [])
            if not models_list:
                logger.warning("未找到 'models' 数组")
            else:
                # 找到第一个 type 为 "llm" 的模型（忽略 embedding）
                llm_model = None
                for m in models_list:
                    if m.get('type') == 'llm':
                        llm_model = m
                        break
                if not llm_model:
                    llm_model = models_list[0]
                    logger.warning("未找到 type='llm' 的模型，使用第一个")
                
                publisher = llm_model.get('publisher')
                display_name = llm_model.get('display_name') or llm_model.get('key')
                quant_info = llm_model.get('quantization')
                quant_name = quant_info.get('name') if quant_info else None
                
                # 获取运行时上下文长度（用户实际设置的值）
                ctx_len = None
                loaded_instances = llm_model.get('loaded_instances', [])
                if loaded_instances:
                    config = loaded_instances[0].get('config', {})
                    ctx_len = config.get('context_length')  # 运行时实际值
                    if ctx_len is None:
                        # 如果 config 中没有，尝试从 loaded_instances 顶层获取（某些版本）
                        ctx_len = loaded_instances[0].get('context_length')
                
                # 如果没有运行时值，则回退到模型最大能力
                if ctx_len is None:
                    ctx_len = llm_model.get('max_context_length')
                    if ctx_len:
                        logger.warning(f"未找到运行时 context_length，使用模型最大能力: {ctx_len}")
                
                # 获取模型实际 ID（用于 API 调用）
                if loaded_instances:
                    model_id = loaded_instances[0].get('id')
                else:
                    model_id = llm_model.get('key')
                if not model_id:
                    model_id = "unknown-model"
                
                logger.info(f"✅ 从 /api/v1/models 获取模型详情: id={model_id}, publisher={publisher}, display={display_name}, quant={quant_name}, 运行时上下文={ctx_len}")
                return model_id, publisher, display_name, quant_name, ctx_len
        else:
            logger.warning(f"/api/v1/models 返回状态码 {resp.status_code}")
    except Exception as e:
        logger.debug(f"访问 /api/v1/models 失败: {e}")

    # 回退到旧的 OpenAI 兼容端点
    logger.info("回退到 /v1/models 端点...")
    try:
        resp = requests.get("http://localhost:1234/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get('data', [])
        if models:
            model_id = models[0].get('id')
            logger.info(f"✅ 从 /v1/models 获取模型: {model_id}")
            # 旧端点没有 publisher/quant 等，返回 None
            return model_id, None, None, None, None
    except Exception as e:
        logger.error(f"无法获取模型列表: {e}")

    return None, None, None, None, None

# ---------- 模型加载检查 ----------
def check_model_loaded(context_length=None, manual_meta=None):
    global ACTUAL_MODEL_NAME, MODEL_CONTEXT_LENGTH, MAX_TOKENS, CHUNK_SIZE
    global MODEL_PUBLISHER, MODEL_DISPLAY_NAME, MODEL_QUANT_NAME

    # 获取模型详情
    model_id, publisher, display_name, quant_name, api_ctx = get_model_details()
    if not model_id:
        logger.error("❌ 无法获取任何模型信息，请确保 LM Studio 正在运行且已加载模型。")
        return False

    ACTUAL_MODEL_NAME = model_id
    MODEL_PUBLISHER = publisher
    MODEL_DISPLAY_NAME = display_name
    MODEL_QUANT_NAME = quant_name

    # 如果用户手动指定了元数据，则覆盖
    if manual_meta:
        parts = manual_meta.split(',', 2)
        if len(parts) >= 1 and parts[0].strip():
            MODEL_PUBLISHER = parts[0].strip()
        if len(parts) >= 2 and parts[1].strip():
            MODEL_DISPLAY_NAME = parts[1].strip()
        if len(parts) >= 3 and parts[2].strip():
            MODEL_QUANT_NAME = parts[2].strip()
        logger.info(f"📝 使用手动指定的模型元数据: 作者={MODEL_PUBLISHER}, 模型={MODEL_DISPLAY_NAME}, 量化={MODEL_QUANT_NAME}")

    # 最终回退
    if not MODEL_PUBLISHER:
        MODEL_PUBLISHER = "unknown_author"
    if not MODEL_DISPLAY_NAME:
        MODEL_DISPLAY_NAME = model_id
    if not MODEL_QUANT_NAME:
        MODEL_QUANT_NAME = "unknown_quant"

    logger.info(f"🏷️ 模型元数据: 发布者={MODEL_PUBLISHER}, 名称={MODEL_DISPLAY_NAME}, 量化={MODEL_QUANT_NAME}")

    # 确定上下文长度
    actual_ctx = api_ctx
    if actual_ctx is not None:
        logger.info(f"📏 模型实际报告的上下文长度: {actual_ctx} tokens")
    else:
        logger.info("📏 模型信息中未提供上下文长度，将根据模型名称推断。")

    if context_length is not None:
        used_ctx = context_length
        logger.info(f"📏 使用手动指定的上下文长度: {used_ctx} tokens")
    else:
        if actual_ctx is not None:
            used_ctx = actual_ctx
            logger.info(f"📏 使用模型报告的长度: {used_ctx} tokens")
        else:
            if 'qwen' in model_id.lower():
                used_ctx = 16384
            else:
                used_ctx = 8192
            logger.info(f"📏 根据模型名称推断上下文长度: {used_ctx} tokens")

    if actual_ctx is not None:
        if used_ctx == actual_ctx:
            logger.info(f"✅ 上下文长度设置一致: {used_ctx} tokens")
        else:
            logger.warning(f"⚠️ 上下文长度不一致：设置值={used_ctx}, 模型实际={actual_ctx}")
            logger.warning(f"   将使用设置值 {used_ctx} tokens 进行分块计算。")
    else:
        logger.info(f"📏 使用推断/指定的上下文长度: {used_ctx} tokens")

    MODEL_CONTEXT_LENGTH = used_ctx

    # 动态计算
    MAX_TOKENS = min(16384, max(1024, MODEL_CONTEXT_LENGTH - 512))
    CHUNK_SIZE = min(8000, max(1000, int((MODEL_CONTEXT_LENGTH - 512) * 4 * 0.8)))

    logger.info(f"🔢 动态 max_tokens: {MAX_TOKENS} tokens")
    logger.info(f"📦 动态 chunk_size: {CHUNK_SIZE} 字符")
    return True

# ---------- 翻译辅助函数 ----------
def clean_translation(text):
    text = re.sub(r'[（\(][^）\)]*名词解释[^）\)]*[）\)]', '', text)
    text = re.sub(r'[（\(][^）\)]*注[：:][^）\)]*[）\)]', '', text)
    text = re.sub(r'\n注[：:].*$', '', text, flags=re.MULTILINE)
    return text.strip()

def translate_text(text, retry=5, profile=0):
    if not text.strip():
        return ""
    if profile < 0 or profile >= len(PROFILES):
        profile = 0
    if MAX_TOKENS is None:
        logger.error("MAX_TOKENS 未初始化，请检查模型加载。")
        return "[错误] " + text

    sys_prompt = PROFILES[profile]["system_prompt"]
    temperature = PROFILES[profile]["temperature"]
    payload = {
        "model": ACTUAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    for attempt in range(retry):
        try:
            global API_CALL_COUNT
            API_CALL_COUNT += 1
            response = requests.post(LM_STUDIO_API, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            response.raise_for_status()
            translated = response.json()['choices'][0]['message']['content'].strip()
            return clean_translation(translated)
        except requests.exceptions.Timeout:
            if attempt == retry - 1:
                logger.error(f"翻译超时，放弃: {text[:50]}...")
                return "[超时] " + text
            wait = 2 ** attempt
            logger.warning(f"超时，{attempt+1}/{retry} 次重试，等待 {wait}s")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_text = e.response.text
                logger.error(f"翻译请求失败 (400): {error_text[:500]}")
                if "Model reloaded" in error_text or "crashed" in error_text:
                    logger.error("模型崩溃或重载，请检查 LM Studio 状态。")
                    logger.error("建议：降低上下文长度（例如 16384）、关闭其他占用资源的程序后重试。")
                    sys.exit(1)
            if "No models loaded" in str(e):
                logger.error("❌ LM Studio 模型未加载，请先加载模型。")
                sys.exit(1)
            if attempt == retry - 1:
                logger.error(f"翻译失败: {e}，原文: {text[:50]}...")
                return "[失败] " + text
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retry - 1:
                logger.error(f"翻译失败: {e}，原文: {text[:50]}...")
                return "[失败] " + text
            time.sleep(2 ** attempt)

# ---------- 动态计算最优批次大小（更保守） ----------
def calculate_optimal_batch_size(texts, profile=0):
    """
    根据模型上下文长度和待翻译文本的预估 token 数，动态计算每批最多包含的段落数。
    返回整数 batch_size (至少 1，最多 12，且不超过 texts 长度)。
    """
    if not texts:
        return 1
    if MODEL_CONTEXT_LENGTH is None:
        return min(6, len(texts))
    
    safety_margin = 512
    available_tokens = MODEL_CONTEXT_LENGTH - safety_margin
    
    overhead_per_segment = 25
    segment_tokens = [max(1, len(seg) // 3) for seg in texts]
    
    total_tokens = sum(segment_tokens) + len(texts) * overhead_per_segment
    if total_tokens <= available_tokens:
        return min(len(texts), 12)  # 最多 12 段
    
    cum_tokens = 0
    batch_size = 0
    for i, seg_token in enumerate(segment_tokens):
        if cum_tokens + seg_token + overhead_per_segment <= available_tokens:
            cum_tokens += seg_token + overhead_per_segment
            batch_size += 1
        else:
            break
    
    batch_size = max(1, min(batch_size, 12, len(texts)))
    return batch_size

# ---------- 批处理翻译（动态批次大小，带重试） ----------
def batch_translate_texts(texts, profile=0, max_workers=DEFAULT_MAX_WORKERS, retry_on_format_error=2):
    if not texts:
        return []
    if len(texts) == 1:
        return [translate_text(texts[0], profile=profile)]

    # 如果段落数大于 8，尝试分批（递归），但批次大小动态计算
    if len(texts) > 8:
        batch_size = calculate_optimal_batch_size(texts, profile)
        # 避免无限递归
        if batch_size >= len(texts):
            # 直接走单次批处理
            pass
        else:
            logger.debug(f"动态批次大小: {batch_size} (总段落数: {len(texts)})")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = batch_translate_texts(batch, profile, max_workers, retry_on_format_error)
                results.extend(batch_results)
            return results

    # 单次批处理逻辑（带格式错误重试）
    sys_prompt = PROFILES[profile]["system_prompt"]
    # 强化系统提示：强制使用分隔符，禁止额外输出
    sys_prompt_text = (
        sys_prompt +
        f"\n\n请将以下 {len(texts)} 个段落分别翻译成中文。\n"
        f"每个段落必须用 '===SEGMENT===' 单独分隔。\n"
        f"请按相同顺序返回翻译结果，同样用 '===SEGMENT===' 分隔。\n"
        f"只输出翻译结果，不要添加任何额外解释、标记或代码块。\n"
        f"不要输出 '```' 或其他格式。\n"
    )
    separator = "\n===SEGMENT===\n"
    merged_input = separator.join(texts)

    payload = {
        "model": ACTUAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt_text},
            {"role": "user", "content": merged_input}
        ],
        "temperature": PROFILES[profile]["temperature"],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    for attempt in range(retry_on_format_error + 1):
        try:
            global API_CALL_COUNT
            API_CALL_COUNT += 1
            response = requests.post(LM_STUDIO_API, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content'].strip()

            # 移除可能的 markdown 代码块标记
            if content.startswith('```'):
                content = re.sub(r'^```\w*\n', '', content)
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # 尝试解析 JSON
            try:
                result = json.loads(content)
                if isinstance(result, list) and len(result) == len(texts):
                    logger.info(f"✅ 批处理成功 (JSON): {len(texts)} 段")
                    return result
            except json.JSONDecodeError:
                pass

            # 尝试分割
            parts = re.split(r'\s*===SEGMENT===\s*', content)
            # 去除可能的首尾空字符串
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == len(texts):
                logger.info(f"✅ 批处理成功 (文本分隔): {len(texts)} 段")
                return parts

            # 格式错误
            logger.warning(f"批处理格式错误 (尝试 {attempt+1}/{retry_on_format_error+1}): 期望 {len(texts)} 段, 实际 {len(parts)} 段")
            if attempt < retry_on_format_error:
                # 重试前稍微降低温度以增加稳定性
                payload["temperature"] = max(0.1, payload["temperature"] - 0.05)
                logger.info(f"降低温度至 {payload['temperature']}，重试批处理...")
                continue
            else:
                logger.debug(f"批处理输出内容前500字符: {content[:500]}")
                break

        except Exception as e:
            logger.debug(f"批处理请求失败 (尝试 {attempt+1}/{retry_on_format_error+1}): {e}")
            if attempt < retry_on_format_error:
                time.sleep(2)
                continue
            else:
                break

    # 所有重试都失败，回退到并发逐个翻译
    logger.info(f"⏪ 批处理失败，改用并发逐个翻译 {len(texts)} 段")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_text, t, 5, profile): i for i, t in enumerate(texts)}
        results = [None] * len(texts)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"翻译失败: {e}")
                results[idx] = "[错误] " + texts[idx]
        return results

# ---------- 其余函数保持不变（split_text_by_punctuation, translate_long_text, analyze_epub_paragraphs,
#            translate_lines_parallel, batch_translate_lines, make_bilingual_soup,
#            safe_replace_body, create_epub_from_dir, is_valid_epub, find_existing_progress,
#            process_epub, main 等） ----------
# 为了节省篇幅，这里省略了后续未修改的函数，但实际使用时需要将它们完整复制过来。
# 下面仅列出必须保留的函数占位，实际请从之前的完整代码中复制。

def split_text_by_punctuation(text, max_chunk):
    separators = re.compile(r'(?<=[。！？；;.!?])\s*')
    segments = [seg.strip() for seg in separators.split(text) if seg.strip()]
    if not segments:
        return []
    out = []
    current = ''
    for seg in segments:
        if len(seg) > max_chunk:
            if current:
                out.append(current.strip())
                current = ''
            for i in range(0, len(seg), max_chunk):
                out.append(seg[i:i+max_chunk].strip())
        else:
            if not current:
                current = seg
            elif len(current) + len(seg) <= max_chunk:
                current += seg
            else:
                out.append(current.strip())
                current = seg
    if current:
        out.append(current.strip())
    return out

def translate_long_text(text, profile=0, chunk_size=None, whole_attempt_limit=None):
    cs = chunk_size if chunk_size is not None else CHUNK_SIZE
    wal = whole_attempt_limit if whole_attempt_limit is not None else MAX_WHOLE_ATTEMPT_LIMIT
    if len(text) <= cs:
        return translate_text(text, profile=profile)
    if len(text) <= wal:
        try:
            logger.debug(f"尝试整段翻译 ({len(text)} 字符)...")
            return translate_text(text, profile=profile)
        except Exception as e:
            logger.warning(f"整段翻译失败，回退到分块翻译: {e}")
    paragraphs = re.split(r'\n\s*\n', text.strip())
    if len(paragraphs) == 1:
        logger.debug(f"段落过长，按字符分块 ({len(text)} 字符)")
        chunks = [text[i:i+cs] for i in range(0, len(text), cs)]
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"分块 {i+1}/{len(chunks)} ({len(chunk)} 字符)")
            translated_chunks.append(translate_text(chunk, profile=profile))
        return '\n'.join(translated_chunks)
    translated_paras = []
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
        if len(para) <= cs:
            logger.debug(f"段落 {i+1}/{len(paragraphs)} ({len(para)} 字符)")
            translated_paras.append(translate_text(para, profile=profile))
        else:
            if len(para) <= wal:
                try:
                    logger.debug(f"段落 {i+1}/{len(paragraphs)} 尝试整段翻译 ({len(para)} 字符)...")
                    translated_paras.append(translate_text(para, profile=profile))
                    continue
                except Exception as e:
                    logger.warning(f"段落整段翻译失败，回退到分块: {e}")
            sentence_chunks = split_text_by_punctuation(para, cs)
            if len(sentence_chunks) > 1:
                sub_translated = []
                for j, sub in enumerate(sentence_chunks):
                    logger.debug(f"句子块 {j+1}/{len(sentence_chunks)} ({len(sub)} 字符)")
                    sub_translated.append(translate_text(sub, profile=profile))
                translated_paras.append('\n'.join(sub_translated))
            else:
                sub_chunks = [para[j:j+cs] for j in range(0, len(para), cs)]
                sub_translated = []
                for j, sub in enumerate(sub_chunks):
                    logger.debug(f"子块 {j+1}/{len(sub_chunks)} ({len(sub)} 字符)")
                    sub_translated.append(translate_text(sub, profile=profile))
                translated_paras.append('\n'.join(sub_translated))
    return '\n\n'.join(translated_paras)

def analyze_epub_paragraphs(soup):
    body = soup.body
    if body is None:
        return CHUNK_SIZE
    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'blockquote']
    paragraphs = []
    for elem in body.find_all(block_tags):
        if elem.get('class') and 'translation' in elem.get('class'):
            continue
        if elem.find(block_tags):
            continue
        if not elem.get_text(strip=True):
            continue
        if elem.name in IGNORE_TAGS:
            continue
        text = elem.get_text(separator=' ').strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        return CHUNK_SIZE
    para_lengths = [len(p) for p in paragraphs]
    total_chars = sum(para_lengths)
    avg_para_length = total_chars / len(paragraphs)
    suggested_chunk = min(math.ceil(avg_para_length * 8), CHUNK_SIZE)
    logger.info(f"📊 文件段落分析: {len(paragraphs)} 段，总 {total_chars} 字符")
    logger.info(f"  - 平均段落长度: {avg_para_length:.1f} 字符")
    logger.info(f"  - 建议分块大小: {suggested_chunk} 字符")
    return suggested_chunk

def translate_lines_parallel(lines, profile, doc_chunk_size, doc_whole_attempt_limit, max_workers, total_lines):
    results = [None] * len(lines)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_long_text, line, profile, doc_chunk_size, doc_whole_attempt_limit): idx
                   for idx, line in enumerate(lines)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"并行翻译失败第{idx}行: {e}")
                results[idx] = "[失败] " + lines[idx]
            completed += 1
            logger.info(f"📊 章节内进度: {completed}/{total_lines} 行")
    return results

def batch_translate_lines(lines, translated_all_lines, profile, doc_chunk_size, doc_whole_attempt_limit,
                          calculated_max_batch_lines, max_workers, total_lines):
    batch_buffer = []
    batch_indices = []
    batch_total_chars = 0
    completed = 0

    def update_progress():
        nonlocal completed
        completed = sum(1 for x in translated_all_lines if x is not None)
        logger.info(f"📊 章节内进度: {completed}/{total_lines} 行")

    for i, line in enumerate(lines):
        if len(line) > CHUNK_SIZE:
            translated_all_lines[i] = translate_long_text(line, profile=profile,
                                                          chunk_size=doc_chunk_size,
                                                          whole_attempt_limit=doc_whole_attempt_limit)
            update_progress()
            continue

        if (batch_total_chars + len(line) > MAX_WHOLE_ATTEMPT_LIMIT) or len(batch_buffer) >= calculated_max_batch_lines:
            if batch_buffer:
                batch_results = batch_translate_texts(batch_buffer, profile=profile, max_workers=max_workers)
                if batch_results and len(batch_results) == len(batch_buffer):
                    for idx, result in zip(batch_indices, batch_results):
                        translated_all_lines[idx] = result
                    logger.info(f"✅ 批量处理成功：{len(batch_buffer)} 行")
                    update_progress()
                else:
                    for idx, line_t in zip(batch_indices, batch_buffer):
                        translated_all_lines[idx] = translate_long_text(line_t, profile=profile,
                                                                         chunk_size=doc_chunk_size,
                                                                         whole_attempt_limit=doc_whole_attempt_limit)
                        update_progress()
            batch_buffer = []
            batch_indices = []
            batch_total_chars = 0
        batch_buffer.append(line)
        batch_indices.append(i)
        batch_total_chars += len(line)

    if batch_buffer:
        batch_results = batch_translate_texts(batch_buffer, profile=profile, max_workers=max_workers)
        if batch_results and len(batch_results) == len(batch_buffer):
            for idx, result in zip(batch_indices, batch_results):
                translated_all_lines[idx] = result
            logger.info(f"✅ 批量处理成功：{len(batch_buffer)} 行")
            update_progress()
        else:
            for idx, line_t in zip(batch_indices, batch_buffer):
                translated_all_lines[idx] = translate_long_text(line_t, profile=profile,
                                                                 chunk_size=doc_chunk_size,
                                                                 whole_attempt_limit=doc_whole_attempt_limit)
                update_progress()

def make_bilingual_soup(soup, profile=0, no_batch=False, max_batch_lines=None, max_workers=DEFAULT_MAX_WORKERS):
    global MODEL_CONTEXT_LENGTH
    body = soup.body
    if body is None:
        return soup

    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'blockquote']
    elements = []
    for elem in body.find_all(block_tags):
        if elem.get('class') and 'translation' in elem.get('class'):
            continue
        if elem.find(block_tags):
            continue
        if not elem.get_text(strip=True):
            continue
        if elem.name in IGNORE_TAGS:
            continue
        elements.append(elem)

    if not elements:
        return soup

    doc_chunk_size = analyze_epub_paragraphs(soup)
    lengths = [len(elem.get_text(strip=False)) for elem in elements]
    max_len = max(lengths) if lengths else 0
    doc_whole_attempt_limit = min(max_len, MAX_WHOLE_ATTEMPT_LIMIT)
    logger.info(f"动态参数: 最长段落 {max_len} 字符，整段尝试上限 {doc_whole_attempt_limit}，分块大小 {doc_chunk_size}")

    AVG_TOKENS_PER_LINE = 30
    if MODEL_CONTEXT_LENGTH is None:
        MODEL_CONTEXT_LENGTH = 16384
    batch_enabled = BATCH_ENABLED and not no_batch
    if max_batch_lines is not None:
        calculated_max_batch_lines = min(max_batch_lines, 20)
    else:
        calculated_max_batch_lines = max(3, int(MODEL_CONTEXT_LENGTH / AVG_TOKENS_PER_LINE / 4))
    batch_status = "启用" if batch_enabled else "禁用"
    logger.info(f"📊 批处理参数: 模型上下文 {MODEL_CONTEXT_LENGTH} tokens, 每行约 {AVG_TOKENS_PER_LINE} tokens, 最大批处理行数 {calculated_max_batch_lines} (批处理 {batch_status})")

    element_lines = []
    all_lines = []
    total_chars = 0
    for elem in elements:
        full_text = elem.get_text(separator='\n')
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        if not lines:
            lines = [elem.get_text(strip=False).strip()]
        element_lines.append(lines)
        all_lines.extend(lines)
        total_chars += sum(len(line) for line in lines)

    translated_all_lines = [None] * len(all_lines)
    avg_line_length = total_chars / len(all_lines) if all_lines else 0
    is_short_content = len(all_lines) <= 50 and avg_line_length < 100
    total_lines = len(all_lines)

    if is_short_content:
        logger.info(f"📊 检测到短内容 ({len(all_lines)} 行, 平均 {avg_line_length:.1f} 字符/行)，优先批量翻译")
        if batch_enabled:
            batch_translate_lines(all_lines, translated_all_lines, profile, doc_chunk_size,
                                  doc_whole_attempt_limit, calculated_max_batch_lines, max_workers, total_lines)
        else:
            logger.info("批量翻译模式未启用，使用并行逐行翻译")
            translated_parallel = translate_lines_parallel(all_lines, profile, doc_chunk_size,
                                                           doc_whole_attempt_limit, max_workers, total_lines)
            for i, text in enumerate(translated_parallel):
                translated_all_lines[i] = text
    else:
        logger.info(f"📊 使用标准批处理模式，共 {len(all_lines)} 行")
        if len(all_lines) == 1:
            translated_all_lines[0] = translate_long_text(all_lines[0], profile=profile,
                                                          chunk_size=doc_chunk_size,
                                                          whole_attempt_limit=doc_whole_attempt_limit)
            logger.info(f"📊 章节内进度: 1/{total_lines} 行")
        else:
            batch_translate_lines(all_lines, translated_all_lines, profile, doc_chunk_size,
                                  doc_whole_attempt_limit, calculated_max_batch_lines, max_workers, total_lines)

    line_index = 0
    for elem, lines in zip(elements, element_lines):
        translated_lines = []
        for _ in lines:
            if line_index < len(translated_all_lines):
                translated_lines.append(translated_all_lines[line_index] or "")
                line_index += 1
        translated_elem = soup.new_tag(elem.name)
        for attr, value in elem.attrs.items():
            if attr == 'id':
                continue
            translated_elem[attr] = value
        existing_classes = elem.get('class', [])
        if isinstance(existing_classes, str):
            existing_classes = existing_classes.split()
        translated_elem['class'] = existing_classes + ['translation']
        if elem.name in ['p', 'div', 'blockquote'] and len(translated_lines) > 1:
            filtered_lines = [line for line in translated_lines if line.strip()]
            for i, line in enumerate(filtered_lines):
                translated_elem.append(NavigableString(line))
                if i < len(filtered_lines) - 1:
                    br = soup.new_tag('br')
                    translated_elem.append(br)
        else:
            translated_text = ' '.join(line for line in translated_lines if line)
            translated_elem.append(NavigableString(translated_text))
        elem.insert_after(translated_elem)
    return soup

def safe_replace_body(original_str, new_body_inner):
    pattern = re.compile(r'(<body[^>]*>)(.*?)(</body>)', re.DOTALL | re.IGNORECASE)
    match = pattern.search(original_str)
    if not match:
        logger.warning("未找到 <body> 标签，直接返回原字符串")
        return original_str
    return original_str[:match.start(1)] + match.group(1) + new_body_inner + match.group(3) + original_str[match.end(3):]

def create_epub_from_dir(source_dir, output_path):
    out_tmp = output_path + '.tmp'
    with zipfile.ZipFile(out_tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        mime_path = os.path.join(source_dir, 'mimetype')
        if os.path.exists(mime_path):
            zout.write(mime_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, source_dir)
                if rel == 'mimetype':
                    continue
                zout.write(full, rel)
    shutil.move(out_tmp, output_path)

def is_valid_epub(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            return 'mimetype' in zf.namelist()
    except (zipfile.BadZipFile, FileNotFoundError):
        return False

def find_existing_progress(input_file, profile_idx):
    input_abs = os.path.abspath(input_file)
    search_dirs = {os.path.dirname(input_abs), os.getcwd()}
    results = []
    for search_dir in search_dirs:
        pattern = os.path.join(search_dir, "*.progress.json")
        for prog in glob.glob(pattern):
            try:
                with open(prog, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data_input = data.get('input_file', '')
                if os.path.abspath(data_input) != input_abs:
                    continue
                if data.get('profile') != profile_idx:
                    continue
                output_epub = data.get('output_file', '')
                if not output_epub.endswith('.epub'):
                    continue
                if not os.path.isabs(output_epub):
                    output_epub = os.path.join(search_dir, output_epub)
                output_epub = os.path.abspath(output_epub)
                temp_epub = output_epub + ".inprogress.epub"
                if not any(output_epub == existing[0] for existing in results):
                    results.append((output_epub, prog, temp_epub))
            except Exception as e:
                logger.warning(f"读取进度文件 {prog} 失败: {e}")
                continue
    return results

def process_epub(input_epub, output_epub, profile=0, auto_resume=False, force_restart=False,
                 no_batch=False, max_batch_lines=None, max_workers=DEFAULT_MAX_WORKERS):
    global API_CALL_COUNT
    if not os.path.exists(input_epub):
        logger.error(f"文件不存在：{input_epub}")
        return

    if ACTUAL_MODEL_NAME is None:
        if not check_model_loaded():
            sys.exit(1)

    start_total_time = time.time()
    logger.info(f"开始处理 {input_epub}")
    logger.info(f"翻译专家: {PROFILES[profile]['name']}")
    logger.info(f"批处理模式: {'启用' if BATCH_ENABLED and not no_batch else '禁用'}")
    logger.info(f"并发线程数: {max_workers}")
    logger.info(f"📊 运行参数:")
    logger.info(f"  - 模型上下文长度: {MODEL_CONTEXT_LENGTH} tokens")
    logger.info(f"  - 最大生成 tokens: {MAX_TOKENS} tokens")
    logger.info(f"  - 分块大小: {CHUNK_SIZE} 字符")
    logger.info(f"  - 整段尝试上限: {MAX_WHOLE_ATTEMPT_LIMIT} 字符")

    progress_file = output_epub + ".progress.json"
    temp_epub_base = output_epub + ".inprogress"
    temp_epub = temp_epub_base + ".epub"

    if force_restart:
        for f in glob.glob(output_epub + ".*"):
            if f.endswith(".progress.json") or f.startswith(output_epub + ".inprogress"):
                try:
                    os.remove(f)
                    logger.info(f"清理 {f}")
                except:
                    pass
        resume_from = 0
    else:
        resume_from = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data_output = os.path.abspath(data.get('output_file', ''))
                    data_input = os.path.abspath(data.get('input_file', ''))
                    if data_output == os.path.abspath(output_epub) and data_input == os.path.abspath(input_epub) and data.get('profile') == profile:
                        resume_from = data.get('processed_count', 0)
                        logger.info(f"检测到未完成的翻译进度，已处理 {resume_from} 个文件。")
                        if not auto_resume:
                            ans = input("是否继续上次的翻译？(y/n, 默认y): ").strip().lower()
                            if ans == 'n':
                                resume_from = -1
                        else:
                            logger.info("自动续译模式已启用，将自动继续。")
                if resume_from == -1:
                    try:
                        os.remove(progress_file)
                        logger.info(f"删除进度文件 {progress_file}")
                    except:
                        pass
                    for f in glob.glob(temp_epub_base + "*"):
                        try:
                            os.remove(f)
                            logger.info(f"删除临时文件 {f}")
                        except:
                            pass
                    resume_from = 0
            except Exception as e:
                logger.warning(f"读取进度文件失败: {e}，将从头开始。")
                resume_from = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        if resume_from > 0:
            if not os.path.exists(temp_epub) or not is_valid_epub(temp_epub):
                logger.error(f"临时文件 {temp_epub} 不存在或无效，无法续译，将从头开始。")
                resume_from = 0
            else:
                logger.info(f"从临时文件 {temp_epub} 恢复...")
                with zipfile.ZipFile(temp_epub, 'r') as zin:
                    zin.extractall(tmpdir)
        if resume_from == 0:
            logger.info("解压原始 EPUB...")
            with zipfile.ZipFile(input_epub, 'r') as zin:
                zin.extractall(tmpdir)

        all_files = []
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, tmpdir)
                all_files.append((rel, full))

        xhtml_exts = {'.xhtml', '.html', '.htm', '.xml'}
        xhtml_files = []
        for rel, full in all_files:
            ext = os.path.splitext(rel)[1].lower()
            if ext in xhtml_exts:
                xhtml_files.append((rel, full))
        xhtml_files.sort(key=lambda x: x[0])
        total = len(xhtml_files)
        logger.info(f"共 {total} 个 XHTML 文件需要处理。")

        if resume_from > 0:
            logger.info(f"跳过前 {resume_from} 个已完成的文件。")
            xhtml_files = xhtml_files[resume_from:]

        processed = resume_from
        for idx, (rel, full) in enumerate(xhtml_files, start=resume_from+1):
            logger.info(f"\n{'━'*50}\n📄 文件进度: [{idx}/{total}] {rel}\n{'━'*50}")
            with open(full, 'r', encoding='utf-8-sig', errors='ignore') as f:
                original = f.read()

            try:
                soup = BeautifulSoup(original, 'lxml-xml')
            except ImportError:
                logger.warning("未安装 lxml，使用内置 html.parser 解析，可能丢失部分 XML 特性")
                soup = BeautifulSoup(original, 'html.parser')
            except Exception:
                soup = BeautifulSoup(original, 'html.parser')

            if soup.body:
                make_bilingual_soup(soup, profile=profile, no_batch=no_batch,
                                    max_batch_lines=max_batch_lines, max_workers=max_workers)
                new_inner = soup.body.encode_contents().decode('utf-8')
                modified = safe_replace_body(original, new_inner)
            else:
                modified = original

            with open(full, 'w', encoding='utf-8') as f:
                f.write(modified)

            processed = idx
            logger.info(f"✅ 文件进度: {idx}/{total} 已完成")

            progress_data = {
                "input_file": input_epub,
                "output_file": output_epub,
                "profile": profile,
                "processed_count": processed,
                "timestamp": datetime.now().isoformat()
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)

            try:
                create_epub_from_dir(tmpdir, temp_epub)
                logger.debug(f"临时保存 {temp_epub}")
            except PermissionError:
                new_temp = f"{temp_epub_base}_{uuid.uuid4().hex[:8]}.epub"
                logger.warning(f"临时文件 {temp_epub} 被占用，改用 {new_temp}")
                create_epub_from_dir(tmpdir, new_temp)

        logger.info("打包最终文件...")
        create_epub_from_dir(tmpdir, output_epub)

        try:
            os.remove(progress_file)
            logger.info(f"删除进度文件 {progress_file}")
        except:
            pass
        pattern = temp_epub_base + "*"
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                logger.debug(f"删除临时文件 {f}")
            except OSError as e:
                logger.warning(f"无法删除临时文件 {f}: {e}")

    duration = time.time() - start_total_time
    mins = int(duration // 60)
    secs = duration % 60
    logger.info(f"完成！输出：{output_epub}")
    logger.info(f"总耗时：{mins} 分 {secs:.1f} 秒")
    logger.info(f"总 API 调用次数：{API_CALL_COUNT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 EPUB 书籍翻译成双语对照版（保留结构和样式），支持不同翻译专家")
    parser.add_argument("input_file", help="输入的 EPUB 文件")
    parser.add_argument("--output", default=None, help="输出文件（默认自动添加时间戳，避免覆盖）")
    parser.add_argument("--profile", type=int, default=None,
                        help="选择翻译专家：0=技术翻译，1=文学翻译，2=社科非虚构翻译。若不指定则列出可用选项并退出。")
    parser.add_argument("--context-length", type=int, default=None,
                        help="手动指定模型的上下文长度（tokens），用于计算批处理行数。若不指定，则从模型信息获取或自动推断。")
    parser.add_argument("--model-meta", type=str, default=None,
                        help="手动指定模型元数据，格式: '作者,模型名,量化名' (例如 'bartowski,Qwen2.5 14B Instruct,Q6_K_L')。若不指定则自动从API获取。")
    parser.add_argument("--resume", "--continue", action="store_true", dest="resume", help="自动续译模式，不询问直接继续上次未完成的翻译")
    parser.add_argument("--force", action="store_true", help="强制重新开始，忽略已有进度")
    parser.add_argument("--no-batch", action="store_true", help="禁用批处理模式，逐个翻译所有文本")
    parser.add_argument("--max-batch-lines", type=int, default=None, help="手动设置批处理最大行数，默认自动计算")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"并发翻译线程数（本地模型建议 1~3，默认 {DEFAULT_MAX_WORKERS}）")
    args = parser.parse_args()

    if args.profile is None:
        print("请指定 --profile 参数选择翻译专家：")
        for i, p in enumerate(PROFILES):
            print(f"  {i}: {p['name']}")
        print("例如：--profile 0 使用技术翻译")
        exit(0)

    if args.profile < 0 or args.profile >= len(PROFILES):
        print(f"错误：profile 必须是 0 到 {len(PROFILES)-1} 之间的整数")
        exit(1)

    if not os.path.exists(args.input_file):
        print(f"[错误] 文件不存在：{args.input_file}")
        exit(1)

    if not check_model_loaded(context_length=args.context_length, manual_meta=args.model_meta):
        sys.exit(1)

    safe_publisher = re.sub(r'[\\/*?:"<>|]', '_', MODEL_PUBLISHER)
    safe_model_name = re.sub(r'[\\/*?:"<>|]', '_', MODEL_DISPLAY_NAME)
    safe_quant = re.sub(r'[\\/*?:"<>|]', '_', MODEL_QUANT_NAME)

    if args.output:
        out_file = args.output
    else:
        progresses = find_existing_progress(args.input_file, args.profile)
        if progresses and not args.force:
            if len(progresses) == 1:
                out_file = progresses[0][0]
                print(f"[续译] 检测到未完成的翻译：{out_file}，将自动继续。")
            else:
                print("检测到多个未完成的翻译进度：")
                for i, (out, prog, _) in enumerate(progresses):
                    try:
                        with open(prog, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            cnt = data.get('processed_count', 0)
                        print(f"  {i+1}: {out} (已处理 {cnt} 个文件)")
                    except:
                        print(f"  {i+1}: {out}")
                print("  r: 重新开始新翻译")
                choice = input("请选择要恢复的进度编号（或输入 'r' 重新开始）: ").strip()
                if choice.lower() == 'r':
                    base, ext = os.path.splitext(args.input_file)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_file = f"{base}_{safe_publisher}_{safe_model_name}_{safe_quant}_{timestamp}{ext}"
                else:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(progresses):
                            out_file = progresses[idx][0]
                        else:
                            print("无效选择，将生成新文件。")
                            base, ext = os.path.splitext(args.input_file)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            out_file = f"{base}_{safe_publisher}_{safe_model_name}_{safe_quant}_{timestamp}{ext}"
                    except:
                        print("无效输入，将生成新文件。")
                        base, ext = os.path.splitext(args.input_file)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_file = f"{base}_{safe_publisher}_{safe_model_name}_{safe_quant}_{timestamp}{ext}"
        else:
            base, ext = os.path.splitext(args.input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = f"{base}_{safe_publisher}_{safe_model_name}_{safe_quant}_{timestamp}{ext}"

    process_epub(args.input_file, out_file, profile=args.profile,
                 auto_resume=args.resume, force_restart=args.force,
                 no_batch=args.no_batch, max_batch_lines=args.max_batch_lines,
                 max_workers=args.max_workers)