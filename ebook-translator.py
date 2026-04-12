import sys
import io
import os
import argparse
from typing import List, Dict, Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, JSONDecodeError
from bs4 import BeautifulSoup, NavigableString

LM_STUDIO_URL = "http://localhost:1234"

PROMPT_STYLES = {
    "general": """You are a professional translator. Translate the following text from {source_lang} to {target_lang} naturally and fluently.
Focus on clarity and readability. Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "technical": """You are a technical translator specializing in software development and programming topics.
Translate accurately. Preserve technical terms, code snippets (like `variable names`, `function names`), symbols (like >, ≹), and proper nouns.
When a technical term has no widely-accepted translation, keep it in English.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "academic": """You are an academic translator specializing in research papers and technical articles.
Use precise, formal language. Translate technical terms accurately and maintain the academic tone.
Include transliteration for proper nouns on first occurrence if helpful.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "literary": """You are a literary translator. Preserve the author's voice, tone, and style.
Translate metaphors and idioms creatively to convey the same feeling in the target language.
Maintain the rhythm and flow of the original text.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "news": """You are a news translator. Write in clear, objective, and informative style.
Prioritize accuracy of facts and terminology. Keep headlines concise.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "business": """You are a business translator. Write in professional, formal yet clear style.
Use appropriate business terminology. Be concise and impactful.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "marketing": """You are a marketing translator. Write in engaging, persuasive, and lively style.
Make the content appealing to target audience. Use natural expressions.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "simple": """You are a translator specializing in simplified Chinese for general audiences.
Use short sentences and common vocabulary. Aim for easy understanding at approximately Grade 6 reading level.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "bilingual": """You are a bilingual translator. Keep English technical terms, code, and symbols in English.
Translate surrounding text naturally. Mix Chinese and English where it improves clarity.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:""",

    "podcast": """You are a translator specializing in converting written text into natural spoken Chinese.
Write as if speaking conversationally. Use natural speech patterns and expressions.
Only output the translated text, nothing else.

Text to translate:
{text}

Translation:"""
}

DEFAULT_PROMPT_STYLE = None

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
    except:
        pass
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
    template = PROMPT_STYLES[actual_style]
    prompt = template.format(source_lang=source_lang, target_lang=target_lang, text=text)
    return estimate_tokens(prompt)

def check_paragraph_length(text: str, max_tokens: int) -> tuple[bool, int]:
    tokens = estimate_tokens(text)
    return tokens <= max_tokens, tokens

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
                "publisher": model.get("publisher", "unknown"),
                "name": model.get("display_name", model.get("key", "unknown")),
                "quantization": quant.get("name", "unknown"),
            }
    except:
        pass
    return {"context_length": 4096, "publisher": "unknown", "name": "unknown", "quantization": "unknown"}

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
    def __init__(self, base_url: str = "http://localhost:1234", timeout: int = 3600, max_response_tokens: int = None, style: str = None):
        self.base_url = base_url
        self.timeout = timeout
        self.style = style
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
        template = PROMPT_STYLES[actual_style]
        prompt = template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text
        )
        prompt_tokens = estimate_tokens(prompt)
        
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": self.max_response_tokens
        }
        
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
                except:
                    pass
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
            raise ConnectionError(f"Cannot connect to LM Studio at {self.base_url}")
        except Timeout:
            raise TimeoutError("Translation request timed out")
        except (HTTPError, RuntimeError):
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")

def split_long_paragraph(text: str, max_tokens: int, estimate_fn) -> List[str]:
    import re
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
    
    def create_chapter_content(self, original_html: bytes, translated_text: str) -> str:
        soup = BeautifulSoup(original_html, 'html.parser')
        
        for p in soup.find_all('p'):
            p.string = translated_text
        
        return str(soup)

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

def can_merge_paragraphs(para: dict) -> bool:
    html_str = para.get('html', '')
    soup = BeautifulSoup(html_str, 'html.parser')
    p = soup.find('p')
    if not p:
        return False
    for child in p.children:
        if hasattr(child, 'name') and child.name and child.name not in ['br']:
            return False
    return True

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
            if elem.find(SKIP_TAGS):
                continue
            text = elem.get_text().strip()
            if text and len(text) > 1:
                blocks.append(elem)
    return blocks

def translate_node(text: str, source_lang: str, target_lang: str, translator, max_para_tokens: int) -> dict:
    tokens = estimate_tokens(text)
    
    if tokens > max_para_tokens:
        chunks = split_long_paragraph(text, max_para_tokens, estimate_tokens)
        results = []
        total_tokens = 0
        total_time = 0.0
        for chunk in chunks:
            try:
                result = translator.translate(chunk, source_lang, target_lang)
                results.append(result["text"])
                total_tokens += result.get("total_tokens", 0)
                total_time += result.get("elapsed", 0.0)
            except Exception as e:
                results.append(f"[Error: {str(e)}]")
        return {
            "original": text,
            "translated": ''.join(results),
            "tokens": tokens,
            "total_tokens": total_tokens,
            "elapsed": total_time,
            "split": True
        }
    else:
        try:
            result = translator.translate(text, source_lang, target_lang)
            return {
                "original": text,
                "translated": result.get("text", ""),
                "tokens": tokens,
                "total_tokens": result.get("total_tokens", 0),
                "elapsed": result.get("elapsed", 0.0),
                "split": False
            }
        except Exception as e:
            return {
                "original": text,
                "translated": f"[Error: {str(e)}]",
                "tokens": tokens,
                "total_tokens": 0,
                "elapsed": 0.0,
                "split": False
            }

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
            empty_p = soup.new_tag('p')
            trans_block = BeautifulSoup(f'<div class="translation">{translated}</div>', 'html.parser')
            block.insert_after(empty_p)
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

def generate_output_filename(original_filename: str, translated_filename: str, model_info: dict, style: str) -> str:
    from datetime import datetime
    import re
    
    safe_original = re.sub(r'[\\/:*?"<>|]', '_', original_filename)
    safe_translated = re.sub(r'[\\/:*?"<>|]', '_', translated_filename)
    if len(safe_translated) > 100:
        safe_translated = safe_translated[:100]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    safe_publisher = model_info["publisher"].replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_name = model_info["name"].replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_quant = model_info["quantization"].replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    base_name = f"{safe_original}_{safe_translated}_{safe_publisher}_{safe_name}_{safe_quant}_{style}_{timestamp}"
    
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
    parser.add_argument('-p', '--parallel', type=int, default=4, help='Number of parallel translation workers (default: 4)')
    parser.add_argument('--prompt-style', default=None,
                        choices=list(PROMPT_STYLES.keys()),
                        help='Translation style (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"[Error] File not found: {args.input_file}")
        sys.exit(1)
    
    if args.output is None:
        model_info = get_model_info(args.lm_url)
    else:
        model_info = None
    
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
        name_result = translator.translate(original_name, args.source, args.target)
        translated_name = name_result.get("text", original_name)
        print(f"  Filename: {original_name} → {translated_name}")
        args.output = generate_output_filename(original_name, translated_name, model_info, args.prompt_style)
    
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