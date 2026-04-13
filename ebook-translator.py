import re
import sys
import io
import os
import argparse
import warnings
from typing import List, Dict, Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
from bs4 import BeautifulSoup, NavigableString

LM_STUDIO_URL = "http://localhost:1234"

PROMPT_STYLES = {
    "general": {
        "prompt": """你是专业翻译专家。将以下{源语言}翻译成自然流畅的{目标语言}。
保持译文清晰易读，人名、地名、数字保留原文格式。
直接输出翻译结果，不要有任何思考过程或额外说明。""",
        "temperature": 0.3
    },

    "technical": {
        "prompt": """你是专业技术书籍翻译专家。将以下英文翻译成准确、流畅、保留技术术语的{目标语言}。
保留代码块和行内代码的原始格式，技术术语保持一致。
直接输出翻译结果，不要有任何思考过程、脚注、括号注释或多余文字。""",
        "temperature": 0.3
    },

    "academic": {
        "prompt": """你是专业学术翻译专家。将以下英文翻译成准确、严谨、流畅的{目标语言}。
使用精确正式的语言，首次出现的人名提供音译（如"Transformer (Transformer)"），保留引用和参考文献的原文格式。
直接输出翻译结果，不要有任何思考过程、脚注或额外说明。""",
        "temperature": 0.3
    },

    "literary": {
        "prompt": """你是专业文学翻译专家，擅长翻译英文小说。将以下英文翻译成优美、自然、保留原著情感和风格的中文。
保留人物的语气、性格和情感色彩，对话翻译需口语化自然，关键意象和隐喻尽量保留。
直接输出翻译结果，不要有任何思考过程、脚注、括号注释或额外说明。""",
        "temperature": 0.4
    },

    "news": {
        "prompt": """你是专业新闻翻译专家。将以下英文翻译成清晰、客观、易读的{目标语言}。
保持事实准确，术语规范，标题简洁。
直接输出翻译结果，不要有任何思考过程或额外说明。""",
        "temperature": 0.3
    },

    "business": {
        "prompt": """你是专业商业翻译专家。将以下英文翻译成专业、正式、清晰、简洁的{目标语言}。
使用恰当的商业术语，保持术语一致性。
直接输出翻译结果，不要有任何思考过程或额外说明。""",
        "temperature": 0.3
    },

    "marketing": {
        "prompt": """你是专业营销内容翻译专家。将以下英文翻译成有吸引力、有说服力、生动活泼的{目标语言}。
保持说服力但忠实于原文信息，使用自然的表达方式。
直接输出翻译结果，不要有任何思考过程或原文没有的夸张修辞。""",
        "temperature": 0.4
    },

    "simple": {
        "prompt": """你是专业翻译专家，擅长为普通读者翻译。将以下英文翻译成简洁易懂的{目标语言}。
使用短句（每句不超过20个词）、常见词汇，避免被动语态。
直接输出翻译结果，不要有任何思考过程或额外说明。""",
        "temperature": 0.3
    },

    "bilingual": {
        "prompt": """你是双语技术翻译专家。将以下英文翻译成自然的中文。
只保留无法翻译的术语为英文（如API、JSON、Git、URL），其他内容全部翻译成中文。如有帮助，可使用"英文术语（中文解释）"格式。
直接输出翻译结果，不要有任何思考过程或此格式外的任何解释。""",
        "temperature": 0.3
    },

    "podcast": {
        "prompt": """你是专业翻译专家，擅长将书面文本转换为自然口语化的中文。
像说话一样自然流畅，使用"和"、"所以"、"但是"等自然连接词，移除正式书面标记。
直接输出翻译结果，不要有任何思考过程或额外说明。""",
        "temperature": 0.4
    }
}

DETECTION_PROMPT = """Read the text sample below and determine the most appropriate translation style.

Choose ONE from:
- technical: software, programming, IT, engineering docs
- academic: research papers, scholarly articles
- literary: novels, stories, poetry, creative writing
- news: journalism, reports
- business: corporate documents, professional writing
- marketing: promotional content, advertisements
- general: everyday content, mixed topics

Text sample:
{text}

Respond with ONLY a single word (the style name):"""

DETECTION_SAMPLE_SIZE = 500

def detect_content_type(text_sample: str, translator) -> str:
    payload = {
        "messages": [
            {"role": "user", "content": DETECTION_PROMPT.format(text=text_sample[:DETECTION_SAMPLE_SIZE])}
        ],
        "temperature": 0.1,
        "max_tokens": 20
    }
    try:
        response = requests.post(
            f"{translator.base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            detected = result["choices"][0]["message"]["content"].strip().lower()
            if detected in PROMPT_STYLES:
                return detected
    except Exception as e:
        warnings.warn(f"Content type detection failed: {e}")
    return "general"

def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()

def estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    result = int(chinese_chars / 1.5 + other_chars / 4)
    return max(1, result) if text.strip() else 0

def get_prompt_style(style: str) -> str:
    return style if style else "general"

def estimate_prompt_tokens(text: str, source_lang: str, target_lang: str, style: str = None) -> int:
    actual_style = get_prompt_style(style)
    system_prompt = PROMPT_STYLES[actual_style]["prompt"].format(源语言=source_lang, 目标语言=target_lang)
    return estimate_tokens(system_prompt + text)

def get_model_info(base_url: str) -> dict:
    try:
        response = requests.get(f"{base_url}/api/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        loaded_models = [m for m in models if m.get("loaded_instances")]
        if loaded_models:
            model = loaded_models[0]
            inst = model["loaded_instances"][0]
            quant = model.get("quantization", {})
            return {
                "context_length": inst.get("config", {}).get("context_length", 4096),
                "parallel": inst.get("config", {}).get("parallel", 1),
                "publisher": model.get("publisher", "unknown"),
                "name": model.get("display_name", model.get("key", "unknown")),
                "quantization": quant.get("name", "unknown"),
            }
    except Exception as e:
        warnings.warn(f"Failed to get model info: {e}")
    return {"context_length": 4096, "parallel": 1, "publisher": "unknown", "name": "unknown", "quantization": "unknown"}

def get_loaded_context_length(base_url: str) -> int:
    return get_model_info(base_url)["context_length"]

def calculate_max_para_tokens(base_url: str, source_lang: str, target_lang: str, style: str = None) -> int:
    context_length = get_loaded_context_length(base_url)
    sample_text = "a" * 100
    prompt_tokens = estimate_prompt_tokens(sample_text, source_lang, target_lang, style)
    reserved = prompt_tokens + int(context_length * 0.1)
    return max(500, context_length - reserved)

class EpubParser:
    def __init__(self):
        try:
            import ebooklib
            from ebooklib import epub
            self.epub = epub
        except ImportError:
            raise ImportError("ebooklib is required for EPUB parsing. Install it with: pip install ebooklib")
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        book = self.epub.read_epub(file_path)
        chapters = []
        
        for item in book.get_items():
            if isinstance(item, self.epub.EpubHtml):
                content = item.get_content()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                soup = BeautifulSoup(content, 'html.parser')
                self._remove_scripts_and_styles(soup)
                chapters.append({
                    'id': item.id,
                    'file_name': getattr(item, 'file_name', item.id),
                    'item': item,
                    'soup': soup
                })
        
        return chapters
    
    def _remove_scripts_and_styles(self, soup):
        for script in soup(['script', 'style']):
            script.decompose()

class LMStudioTranslator:
    def __init__(self, base_url: str = "http://localhost:1234", timeout: int = 3600, max_response_tokens: int = None, style: str = None, max_retries: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.style = style
        self.max_retries = max_retries
        self.total_tokens = 0
        self.total_time = 0.0
        if max_response_tokens is None:
            context_length = get_loaded_context_length(base_url)
            self.max_response_tokens = int(context_length * 0.4)
        else:
            self.max_response_tokens = max_response_tokens
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        import time
        actual_style = get_prompt_style(self.style)
        style_config = PROMPT_STYLES[actual_style]
        system_prompt = style_config["prompt"].format(
            源语言=source_lang,
            目标语言=target_lang
        )
        
        prompt_tokens = estimate_tokens(system_prompt + text)
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": style_config["temperature"],
            "max_tokens": self.max_response_tokens
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"].get("message", str(error_data["error"]))
                            if "no models loaded" in error_msg.lower():
                                raise RuntimeError(f"LM Studio error: {error_msg}\n\nPlease load a model in LM Studio first!")
                            raise RuntimeError(f"LM Studio error: {error_msg}")
                    except RuntimeError:
                        raise
                    except Exception as e:
                        warnings.warn(f"Failed to parse error response: {e}")
                    raise HTTPError(f"HTTP error: {response.status_code}")
                
                data = response.json()
                
                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise RuntimeError(f"LM Studio error: {error_msg}")
                
                elapsed = time.time() - start_time
                
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                
                self.total_tokens += total_tokens
                self.total_time += elapsed
                
                return {
                    "text": data["choices"][0]["message"]["content"].strip(),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "elapsed": elapsed
                }
            except requests.exceptions.ConnectionError:
                last_error = ConnectionError(f"Cannot connect to LM Studio at {self.base_url}")
                warnings.warn(f"Attempt {attempt + 1}/{self.max_retries} failed: {last_error}")
                continue
            except Timeout:
                last_error = TimeoutError("Translation request timed out")
                warnings.warn(f"Attempt {attempt + 1}/{self.max_retries} failed: {last_error}")
                continue
            except (HTTPError, RuntimeError):
                raise
            except Exception as e:
                last_error = ValueError(f"Unexpected error: {str(e)}")
                warnings.warn(f"Attempt {attempt + 1}/{self.max_retries} failed: {last_error}")
                continue
        
        raise last_error

def split_long_paragraph(text: str, max_tokens: int, estimate_fn) -> List[str]:
    sentence_end = re.compile(r'[。！？.!?]+')
    sentences = []
    current = []
    current_tokens = 0
    
    for char in text:
        current.append(char)
        if sentence_end.match(char):
            sentence = ''.join(current)
            tokens = estimate_fn(sentence)
            if current_tokens + tokens > max_tokens and current_tokens > 0:
                sentences.append(''.join(current[:-len(char)]))
                current = [char]
                current_tokens = tokens
            else:
                current_tokens += tokens
            current = []
    
    if current:
        remaining = ''.join(current)
        if remaining.strip():
            sentences.append(remaining)
    
    return sentences

class EpubExporter:
    def __init__(self):
        try:
            from ebooklib import epub
            self.epub = epub
        except ImportError:
            raise ImportError("ebooklib is required for EPUB export. Install it with: pip install ebooklib")
    
    def export(self, chapters: List[Dict[str, Any]], output_path: str, original_file: str) -> None:
        import tempfile
        import shutil
        import zipfile
        
        print(f"  Copying original epub...", flush=True)
        shutil.copy2(original_file, output_path)
        
        print(f"  Extracting to temp dir...", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(output_path, 'r') as zf:
                zf.extractall(tmpdir)
            
            for chapter in chapters:
                file_name = chapter.get('file_name', chapter['id']).replace('/', os.sep)
                
                file_path = None
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        full_path = os.path.join(root, f)
                        if f == file_name or full_path.endswith(file_name) or file_name in full_path:
                            file_path = full_path
                            break
                    if file_path:
                        break
                
                if not file_path or not os.path.exists(file_path):
                    print(f"  WARNING: Cannot find file: {file_name}", flush=True)
                    continue
                
                soup = chapter['soup']
                modified_content = str(soup)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                print(f"  Modified: {file_name} ({len(modified_content)} bytes)", flush=True)
            
            print(f"  Repacking epub...", flush=True)
            out_tmp = output_path + '.tmp'
            with zipfile.ZipFile(out_tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arc_name = os.path.relpath(full_path, tmpdir)
                        zout.write(full_path, arc_name)
            
            shutil.move(out_tmp, output_path)
        
        print(f"  Export complete", flush=True)

def parse_ebook(file_path: str) -> List[Dict[str, Any]]:
    ext = get_file_extension(file_path)
    
    if ext != '.epub':
        raise ValueError(f"Only EPUB format is supported. Got: {ext}")
    
    parser = EpubParser()
    return parser.parse(file_path)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def collect_text_nodes(soup) -> list:
    SKIP_TAGS = ['script', 'style', 'head', 'title', 'meta', 'link', 'code', 'pre', 'kbd', 'samp', 'tt']
    SKIP_PARENTS = ['[document]', 'html', 'body']
    nodes = []
    for element in soup.find_all(string=True):
        if element.parent.name in SKIP_TAGS:
            continue
        if element.parent.name in SKIP_PARENTS:
            continue
        text = element.strip()
        if text and len(text) > 1:
            nodes.append(element)
    return nodes

def collect_block_elements(soup) -> list:
    BLOCK_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'div', 'span', 'article', 'section', 'blockquote']
    SKIP_TAGS = ['script', 'style', 'head', 'title', 'meta', 'link', 'code', 'pre', 'kbd', 'samp', 'tt']
    
    blocks = []
    for tag in BLOCK_TAGS:
        for elem in soup.find_all(tag):
            if elem.find_parent(SKIP_TAGS):
                continue
            if elem.find(BLOCK_TAGS):
                continue
            text = elem.get_text().strip()
            if text and len(text) > 1:
                blocks.append(elem)
    return blocks

def translate_soup_sequential(soup, translator, source_lang: str, target_lang: str, max_para_tokens: int) -> None:
    blocks = collect_block_elements(soup)
    total_blocks = len(blocks)
    print(f"  {total_blocks} blocks to translate (sequential)")
    
    for i, block in enumerate(blocks):
        text = block.get_text().strip()
        if not text:
            continue
        
        tokens = estimate_tokens(text)
        print(f"    [{i+1}/{total_blocks}] {tokens} tokens...", end=" ", flush=True)
        
        if tokens > max_para_tokens:
            chunks = split_long_paragraph(text, max_para_tokens, estimate_tokens)
            print(f"\n    Split into {len(chunks)} parts")
            results = []
            for k, chunk in enumerate(chunks):
                try:
                    result = translator.translate(chunk, source_lang, target_lang)
                    results.append(result["text"])
                    print(f"      Part {k+1}: {result['total_tokens']} tokens, {format_time(result['elapsed'])}")
                except Exception as e:
                    results.append(f"[Error: {str(e)}]")
                    print(f"      Part {k+1}: FAILED: {e}")
            translated = ''.join(results)
        else:
            try:
                result = translator.translate(text, source_lang, target_lang)
                translated = result.get("text", "")
                print(f"OK, {result.get('total_tokens', 0)} tokens, {format_time(result.get('elapsed', 0))}")
                
                if "no models loaded" in translated.lower():
                    print(f"\n[ERROR] No model loaded in LM Studio. Please load a model and try again.")
                    raise SystemExit(1)
            except SystemExit:
                raise
            except Exception as e:
                print(f"FAILED: {e}")
                translated = f"[Error: {str(e)}]"
        
        try:
            trans_block = BeautifulSoup(f'<div class="ebook-trans-t" style="margin: 0.2em 0 0.3em 0; color: #555;">{translated}</div>', 'html.parser')
            block.insert_after(trans_block)
        except Exception as e:
            print(f"Insert failed: {e}")

def translate_soup(soup, translator, source_lang: str, target_lang: str, max_para_tokens: int) -> None:
    return translate_soup_sequential(soup, translator, source_lang, target_lang, max_para_tokens)

def translate_chapters(chapters: List[Dict[str, Any]], source_lang: str, target_lang: str, translator: LMStudioTranslator, max_para_tokens: int) -> None:
    total_chapters = len(chapters)
    
    for i, chapter in enumerate(chapters):
        print(f"\n  Chapter {i + 1}/{total_chapters}:")
        soup = chapter['soup']
        translate_soup(soup, translator, source_lang, target_lang, max_para_tokens)

def generate_output_filename(original_filename: str, target_lang: str, model_info: dict, style: str) -> str:
    from datetime import datetime
    import re
    
    safe_original = re.sub(r'[\\/:*?"<>|]', '_', original_filename)
    safe_target = re.sub(r'[\\/:*?"<>|]', '_', target_lang)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    safe_publisher = model_info["publisher"].replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_name = model_info["name"].replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_quant = model_info["quantization"].replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    base_name = f"{safe_original}_{safe_target}_{safe_publisher}_{safe_name}_{safe_quant}_{style}_{timestamp}"
    
    output_path = base_name + ".epub"
    seq = 1
    while os.path.exists(output_path):
        seq_str = f"{seq:02d}"
        output_path = f"{base_name}_{seq_str}.epub"
        seq += 1
    
    return output_path

def save_translated_ebook(chapters: List[Dict[str, Any]], output_path: str, original_file: str):
    ext = get_file_extension(output_path)
    
    if ext == '.epub':
        exporter = EpubExporter()
    else:
        raise ValueError(f"Unsupported output format: {ext}")
    
    exporter.export(chapters, output_path, original_file)
    print(f"  Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Ebook Translator using LM Studio')
    parser.add_argument('input_file', help='Input EPUB ebook file')
    parser.add_argument('-o', '--output', help='Output file path (default: input_file.translated.ext)')
    parser.add_argument('-s', '--source', default='auto', help='Source language (default: auto)')
    parser.add_argument('-t', '--target', default='Chinese', help='Target language (default: Chinese)')
    parser.add_argument('--lm-url', default=LM_STUDIO_URL, help=f'LM Studio API URL (default: {LM_STUDIO_URL})')
    parser.add_argument('--timeout', type=int, default=3600, help='Translation timeout in seconds (default: 3600)')
    parser.add_argument('--prompt-style', default=None,
                        choices=list(PROMPT_STYLES.keys()),
                        help='Translation style (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"[Error] File not found: {args.input_file}")
        sys.exit(1)
    
    model_info = get_model_info(args.lm_url)
    
    print("=" * 50)
    print("       Ebook Translator")
    print("=" * 50)
    print(f"  Input:    {args.input_file}")
    if model_info:
        print(f"  Model:    {model_info['publisher']} / {model_info['name']} ({model_info['quantization']})")
    print(f"  Source:   {args.source}")
    print(f"  Target:   {args.target}")
    print(f"  LM URL:   {args.lm_url}")
    print("=" * 50)
    
    print("\n[1/4] Parsing ebook...")
    try:
        chapters = parse_ebook(args.input_file)
    except Exception as e:
        print(f"  [Error] Failed to parse ebook: {e}")
        sys.exit(1)
    
    total_text_nodes = sum(len(collect_text_nodes(c['soup'])) for c in chapters)
    print(f"  Found {len(chapters)} chapters, {total_text_nodes} text nodes")
    
    if args.prompt_style is None:
        print("\n[2/4] Detecting content type...")
        sample_text = chapters[0]['soup'].get_text()[:1000] if chapters else ""
        detector = LMStudioTranslator(base_url=args.lm_url, timeout=30)
        detected_style = detect_content_type(sample_text, detector)
        args.prompt_style = detected_style
        print(f"  Detected style: {detected_style}")
    else:
        print(f"\n[2/4] Translating content (style: {args.prompt_style})...")
    
    context_length = model_info["context_length"]
    max_para_tokens = calculate_max_para_tokens(args.lm_url, args.source, args.target, args.prompt_style)
    print(f"  Model context: {context_length}, max paragraph: {max_para_tokens} tokens")
    
    translator = LMStudioTranslator(base_url=args.lm_url, timeout=args.timeout, style=args.prompt_style)
    
    if args.output is None:
        print("  Translating filename...")
        original_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = generate_output_filename(original_name, args.target, model_info, args.prompt_style)
    
    print(f"  Output:   {args.output}")
    
    print("\n[3/4] Translating chapters...")
    translate_chapters(chapters, args.source, args.target, translator, max_para_tokens)
    
    print("\n[4/4] Saving translated ebook...")
    try:
        save_translated_ebook(chapters, args.output, args.input_file)
    except Exception as e:
        print(f"  [Error] Failed to save ebook: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("  Translation completed!")
    print("=" * 50)
    print(f"  Total tokens: {translator.total_tokens:,}")
    print(f"  Total time:   {format_time(translator.total_time)}")
    print("=" * 50)

if __name__ == "__main__":
    main()