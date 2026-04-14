import re
import sys
import io
import os
import json
import argparse
import shutil
import warnings
from typing import List, Dict, Any, Tuple
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
from bs4 import BeautifulSoup, NavigableString, Comment

LM_STUDIO_URL = "http://localhost:1234"

PROMPT_STYLES = {
    "general": {
        "prompt": """你是专业翻译专家。将以下{源语言}翻译成自然流畅的{目标语言}。
保持译文清晰易读，人名、地名、数字保留原文格式。
保留原文的换行位置和格式，不要将多行文本合并成一行。
直接输出翻译结果，不要有任何思考过程或额外说明。""",
        "temperature": 0.3
    },

    "technical": {
        "prompt": """你是专业技术书籍翻译专家。将以下英文翻译成准确、流畅、保留技术术语的{目标语言}。

【重要】代码符号必须原样保留，包括但不限于：
= > < -> => <- -> :: /**/ -- // /* */ [ ] {{ }} ( ) . , : ; ' " ` ~ ! @ # $ % ^ & * + - = | \\ /

保留所有代码块（<code>...</code>、<pre>...</pre>）和行内代码的原始内容。
保留原文的换行位置和格式，不要将多行文本合并成一行。
技术术语保持一致，不要翻译变量名、函数名、类名、API 名称、协议名称等。
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
        "prompt": """你是专业文学翻译专家，擅长翻译英文小说。将以下英文翻译成优美，自然、保留原著情感和风格的中文。
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

class ProgressState:
    def __init__(self, input_file: str):
        self.input_file = os.path.abspath(input_file)
        self.progress_file = self.input_file + '.progress.json'
        self.data = {
            "input_file": self.input_file,
            "output_file": None,
            "inprogress_file": None,
            "profile": None,
            "target_lang": None,
            "processed_chapters": [],
            "total_chapters": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    def load(self) -> bool:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                return True
            except Exception:
                return False
        return False
    
    def save(self) -> None:
        self.data["timestamp"] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def delete(self) -> None:
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
    
    def is_chapter_processed(self, chapter_file_name: str) -> bool:
        return chapter_file_name in self.data.get("processed_chapters", [])
    
    def mark_chapter_processed(self, chapter_file_name: str) -> None:
        if "processed_chapters" not in self.data:
            self.data["processed_chapters"] = []
        if chapter_file_name not in self.data["processed_chapters"]:
            self.data["processed_chapters"].append(chapter_file_name)
        self.save()
    
    def set_output(self, output_file: str, inprogress_file: str, profile: str, target_lang: str) -> None:
        self.data["output_file"] = output_file
        self.data["inprogress_file"] = inprogress_file
        self.data["profile"] = profile
        self.data["target_lang"] = target_lang
        self.save()
    
    def add_tokens(self, tokens: int) -> None:
        self.data["total_tokens"] = self.data.get("total_tokens", 0) + tokens
    
    def add_time(self, time_seconds: float) -> None:
        self.data["total_time"] = self.data.get("total_time", 0.0) + time_seconds

PROMPT_LEAK_PATTERNS = [
    "代码符号必须原样保留",
    "代码块和行内代码的原始内容",
    "直接输出翻译结果",
    "不要有任何思考过程",
    "technical prompt",
    "system prompt",
]

def cleanup_translation(text: str) -> str:
    text = text.strip()
    
    for pattern in PROMPT_LEAK_PATTERNS:
        idx = text.find(pattern)
        if idx > 0:
            text = text[:idx]
            text = text.strip()
    
    return text

def check_inprogress(input_file: str) -> Tuple[bool, str, str]:
    progress_file = input_file + '.progress.json'
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            inprogress_file = data.get("inprogress_file")
            if inprogress_file and os.path.exists(inprogress_file):
                return True, inprogress_file, progress_file
        except Exception:
            pass
    inprogress_file = os.path.splitext(input_file)[0] + '.inprogress.epub'
    if os.path.exists(inprogress_file):
        return True, inprogress_file, progress_file
    return False, "", ""

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

class InlineProtector:
    def __init__(self):
        self.code_count = 0
        self.link_count = 0
        self.styled_span_count = 0
        self.mappings = {
            'code': {},
            'link': {},
            'styled_span': {}
        }
    
    def protect(self, soup) -> None:
        self._protect_code(soup)
        self._protect_links(soup)
        self._protect_styled_spans(soup)
    
    def _protect_code(self, soup) -> None:
        for code in soup.find_all(['code', 'pre']):
            placeholder = f"___PB_{self.code_count}___"
            self.mappings['code'][placeholder] = code.string if code.string else ''
            code.string = placeholder
            self.code_count += 1
    
    def _protect_links(self, soup) -> None:
        for link in soup.find_all('a'):
            if not link.get('href'):
                continue
            placeholder = f"___PL_{self.link_count}___"
            self.mappings['link'][placeholder] = {
                'text': link.string if link.string else '',
                'href': link.get('href', '')
            }
            link.string = placeholder
            self.link_count += 1
    
    def _protect_styled_spans(self, soup) -> None:
        for span in soup.find_all('span'):
            classes = span.get('class', [])
            if classes and any(c in classes for c in ['italic', 'bold', 'emphasis', 'strong']):
                placeholder = f"___PS_{self.styled_span_count}___"
                self.mappings['styled_span'][placeholder] = {
                    'content': span.string if span.string else '',
                    'class': ' '.join(classes)
                }
                span.string = placeholder
                self.styled_span_count += 1

    def restore(self, soup, translated_text: str) -> str:
        for placeholder, content in self.mappings['code'].items():
            translated_text = translated_text.replace(placeholder, content)
        
        for placeholder, data in self.mappings['link'].items():
            translated_text = translated_text.replace(placeholder, data['text'])
        
        for placeholder, data in self.mappings['styled_span'].items():
            translated_text = translated_text.replace(placeholder, data['content'])
        
        return translated_text
    
    def restore_soup(self, soup) -> None:
        for placeholder, content in self.mappings['code'].items():
            for code in soup.find_all(['code', 'pre']):
                if code.string == placeholder:
                    code.string = content
                    break
        
        for placeholder, data in self.mappings['link'].items():
            for link in soup.find_all('a'):
                if link.string == placeholder:
                    link.string = data['text']
                    break
        
        for placeholder, data in self.mappings['styled_span'].items():
            for span in soup.find_all('span'):
                if span.string == placeholder:
                    span.string = data['content']
                    break

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
                if self._is_nav_or_toc(item):
                    continue
                
                content = item.get_content()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                original_head = None
                original_html_tag = None
                original_body_tag = None
                html_match = re.match(r'(<html[^>]*>)', content)
                if html_match:
                    original_html_tag = html_match.group(1)
                head_match = re.search(r'(<head[^>]*>.*?</head>)', content, re.DOTALL)
                if head_match:
                    original_head = head_match.group(1)
                body_match = re.search(r'(<body[^>]*>)', content)
                if body_match:
                    original_body_tag = body_match.group(1)
                
                soup = BeautifulSoup(content, 'lxml')
                self._remove_scripts_and_styles(soup)
                
                chapters.append({
                    'id': item.id,
                    'file_name': getattr(item, 'file_name', item.id),
                    'item': item,
                    'soup': soup,
                    'protector': InlineProtector(),
                    'original_head': original_head,
                    'original_html_tag': original_html_tag,
                    'original_body_tag': original_body_tag
                })
        
        return chapters
    
    def _is_nav_or_toc(self, item) -> bool:
        file_name = item.file_name.lower() if hasattr(item, 'file_name') else ''
        if 'nav' in file_name:
            return True
        if 'toc' in file_name:
            return True
        
        if hasattr(item, 'attributes'):
            epub_type = item.attributes.get('epub:type', '')
            role = item.attributes.get('role', '')
        else:
            epub_type = getattr(item, 'epub_type', '')
            role = getattr(item, 'role', '')
        
        if epub_type and ('toc' in epub_type.lower() or 'navigation' in epub_type.lower()):
            return True
        
        if role and ('toc' in role.lower() or 'navigation' in role.lower()):
            return True
        
        return False
    
    def _remove_scripts_and_styles(self, soup):
        for element in soup(['script', 'style']):
            element.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

class LMStudioTranslator:
    def __init__(self, base_url: str = "http://localhost:1234", timeout: int = 3600, max_response_tokens: int = None, style: str = None, max_retries: int = 1):
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
        for attempt in range(1):
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
                
                raw_text = data["choices"][0]["message"]["content"].strip()
                cleaned_text = cleanup_translation(raw_text)
                
                return {
                    "text": cleaned_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "elapsed": elapsed
                }
            except requests.exceptions.ConnectionError:
                last_error = ConnectionError(f"Cannot connect to LM Studio at {self.base_url}")
                raise
            except Timeout:
                last_error = TimeoutError("Translation request timed out")
                raise
            except (HTTPError, RuntimeError):
                raise
            except Exception as e:
                last_error = ValueError(f"Unexpected error: {str(e)}")
                raise
        
        raise last_error

def split_long_paragraph(text: str, max_tokens: int, estimate_fn) -> List[str]:
    chinese_punctuation = re.compile(r'[。！？；\u4e00-\u9fff]')
    english_punctuation = re.compile(r'[.!?;]+')
    
    sentences = []
    current = []
    current_tokens = 0
    
    i = 0
    while i < len(text):
        char = text[i]
        current.append(char)
        
        is_punct = False
        if chinese_punctuation.match(char):
            is_punct = True
        elif english_punctuation.match(char):
            is_punct = True
        
        if is_punct:
            sentence = ''.join(current)
            tokens = estimate_fn(sentence)
            
            if current_tokens + tokens > max_tokens and current_tokens > 0:
                if len(sentences) > 0:
                    sentences[-1] = sentences[-1] + ''.join(current[:-len(char)])
                else:
                    sentences.append(''.join(current[:-len(char)]))
                current = [char]
                current_tokens = tokens
            else:
                current_tokens += tokens
            sentences.append(sentence)
            current = []
            current_tokens = 0
        else:
            current_tokens = estimate_fn(''.join(current))
        
        i += 1
    
    if current and ''.join(current).strip():
        remaining = ''.join(current)
        tokens = estimate_fn(remaining)
        
        if sentences and current_tokens + estimate_fn(sentences[-1]) <= max_tokens * 1.5:
            sentences[-1] = sentences[-1] + remaining
        else:
            sentences.append(remaining)
    
    result = []
    for sent in sentences:
        sent_tokens = estimate_fn(sent)
        if sent_tokens > max_tokens:
            sub_chunks = split_by_whitespace_or_marks(sent, max_tokens, estimate_fn)
            result.extend(sub_chunks)
        else:
            result.append(sent)
    
    return result

def split_by_whitespace_or_marks(text: str, max_tokens: int, estimate_fn) -> List[str]:
    chunks = []
    current = []
    current_tokens = 0
    
    for word in re.split(r'(\s+)', text):
        word_tokens = estimate_fn(word)
        if current_tokens + word_tokens > max_tokens and current:
            chunks.append(''.join(current))
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens
    
    if current:
        chunks.append(''.join(current))
    
    return chunks

class EpubExporter:
    def __init__(self):
        try:
            from ebooklib import epub
            self.epub = epub
        except ImportError:
            raise ImportError("ebooklib is required for EPUB export. Install it with: pip install ebooklib")
    
    def export(self, chapters: List[Dict[str, Any]], output_path: str, original_file: str) -> None:
        import tempfile
        import zipfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            if os.path.exists(output_path):
                with zipfile.ZipFile(output_path, 'r') as zf:
                    zf.extractall(tmpdir)
            else:
                shutil.copy2(original_file, output_path)
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
                    continue
                
                soup = chapter['soup']
                original_head = chapter.get('original_head')
                original_html_tag = chapter.get('original_html_tag')
                original_body_tag = chapter.get('original_body_tag')
                
                if original_head and original_html_tag and original_body_tag:
                    body_content = soup.body.encode_contents().decode('utf-8') if soup.body else ''
                    body_tag_match = re.match(r'(<body[^>]*>)', original_body_tag)
                    body_tag_open = body_tag_match.group(1) if body_tag_match else '<body>'
                    modified_content = f'{original_html_tag}\n{original_head}\n{body_tag_open}{body_content}</body>\n</html>'
                else:
                    modified_content = str(soup)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            
            out_tmp = output_path + '.tmp'
            with zipfile.ZipFile(out_tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arc_name = os.path.relpath(full_path, tmpdir)
                        zout.write(full_path, arc_name)
            
            shutil.move(out_tmp, output_path)

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
    SKIP_TAGS = ['script', 'style', 'head', 'title', 'meta', 'link', 'pre', 'kbd', 'samp', 'tt', 'a', 'aside', 'code']
    
    def is_placeholder_only(text):
        return bool(re.match(r'^(\s*___PB_\d+___\s*)+$', text)) or \
               bool(re.match(r'^(\s*___PL_\d+___\s*)+$', text)) or \
               bool(re.match(r'^(\s*___PS_\d+___\s*)+$', text))
    
    blocks = []
    for tag in BLOCK_TAGS:
        for elem in soup.find_all(tag):
            if elem.find_parent(SKIP_TAGS):
                continue
            if elem.find(BLOCK_TAGS):
                continue
            
            if tag == 'p':
                code_children = elem.find_all('code', recursive=False)
                if len(code_children) > 1:
                    for code in code_children:
                        code_text = code.get_text().strip()
                        if code_text and not is_placeholder_only(code_text):
                            blocks.append(code)
                    continue
            
            text = elem.get_text().strip()
            if text and len(text) > 1:
                if is_placeholder_only(text):
                    continue
                blocks.append(elem)
    return blocks

def collect_tables(soup) -> list:
    tables = []
    for table in soup.find_all('table'):
        if table.find_parent('table'):
            continue
        caption = table.find('caption')
        rows = table.find_all('tr')
        if rows:
            tables.append({
                'table': table,
                'caption': caption.get_text().strip() if caption else '',
                'row_count': len(rows)
            })
    return tables

def translate_table(table_data: dict, translator, source_lang: str, target_lang: str) -> str:
    table = table_data['table']
    caption = table_data['caption']
    
    rows = table.find_all('tr')
    if not rows:
        return ''
    
    header_row = rows[0]
    headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
    data_rows = rows[1:]
    
    header_text = ' | '.join(headers) if headers else ''
    
    full_text = header_text
    for row in data_rows:
        cells = [td.get_text().strip() for td in row.find_all(['th', 'td'])]
        full_text += '\n' + ' | '.join(cells)
    
    result = translator.translate(full_text, source_lang, target_lang)
    return result['text']

def translate_soup_sequential(soup, translator, source_lang: str, target_lang: str, max_para_tokens: int, protector: InlineProtector) -> None:
    protector.protect(soup)
    
    blocks = collect_block_elements(soup)
    total_blocks = len(blocks)
    print(f"  {total_blocks} blocks to translate (sequential)")
    
    translations = []
    
    for i, block in enumerate(blocks):
        text = block.get_text().strip()
        if not text:
            continue
        
        if re.match(r'^(\s*___PB_\d+___\s*)+$', text) or re.match(r'^(\s*___PL_\d+___\s*)+$', text) or re.match(r'^(\s*___PS_\d+___\s*)+$', text):
            print(f"    [{i+1}/{total_blocks}] 代码块跳过")
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
                    restored = protector.restore(soup, result["text"])
                    results.append(restored)
                    print(f"      Part {k+1}: {result['total_tokens']} tokens, {format_time(result['elapsed'])}")
                except Exception as e:
                    results.append(f"[Error: {str(e)}]")
                    print(f"      Part {k+1}: FAILED: {e}")
                    raise
            translated = ''.join(results)
        else:
            try:
                result = translator.translate(text, source_lang, target_lang)
                translated = protector.restore(soup, result["text"])
                print(f"OK, {result.get('total_tokens', 0)} tokens, {format_time(result.get('elapsed', 0))}")
                
                if "no models loaded" in translated.lower():
                    print(f"\n[ERROR] No model loaded in LM Studio. Please load a model and try again.")
                    raise SystemExit(1)
            except SystemExit:
                raise
            except Exception as e:
                print(f"FAILED: {e}")
                raise
        
        translations.append((block, translated))
    
    print(f"  Replacing original content with translations...")
    for block, translated in translations:
        try:
            translated_with_br = translated.replace('\n', '<br/>')
            trans_div = BeautifulSoup(
                f'<div class="ebook-trans" style="margin: 0.2em 0 0.3em 0; color: #555;">{translated_with_br}</div>',
                'html.parser'
            ).div
            
            insert_target = block.parent
            if insert_target and insert_target.name in ['blockquote', 'aside', 'article', 'section']:
                insert_target.insert_after(trans_div)
            else:
                block.insert_after(trans_div)
        except Exception as e:
            print(f"  Insert failed: {e}")
    
    print(f"  Restoring inline elements in original blocks...")
    protector.restore_soup(soup)

def translate_soup(soup, translator, source_lang: str, target_lang: str, max_para_tokens: int, protector: InlineProtector) -> None:
    return translate_soup_sequential(soup, translator, source_lang, target_lang, max_para_tokens, protector)

def translate_chapters(chapters: List[Dict[str, Any]], source_lang: str, target_lang: str, translator: LMStudioTranslator, max_para_tokens: int, progress: ProgressState, exporter: EpubExporter, original_file: str) -> None:
    total_chapters = len(chapters)
    completed_count = 0
    
    for i, chapter in enumerate(chapters):
        chapter_file_name = chapter.get('file_name', chapter.get('id', ''))
        
        if progress.is_chapter_processed(chapter_file_name):
            print(f"\n  Chapter {i + 1}/{total_chapters}: [{chapter_file_name}] 已翻译，跳过", flush=True)
            completed_count += 1
            continue
        
        print(f"\n  Chapter {i + 1}/{total_chapters}: [{chapter_file_name}]", flush=True)
        soup = chapter['soup']
        protector = chapter.get('protector', InlineProtector())
        try:
            translate_soup(soup, translator, source_lang, target_lang, max_para_tokens, protector)
        except Exception as e:
            print(f"\n  [ERROR] 翻译失败: {e}")
            print(f"  请检查 LM Studio 是否运行并加载了模型")
            sys.exit(1)
        
        progress.mark_chapter_processed(chapter_file_name)
        progress.add_tokens(translator.total_tokens)
        progress.add_time(translator.total_time)
        completed_count += 1
        
        exporter.export([chapter], progress.data["inprogress_file"], original_file)
        print(f"  进度: {completed_count}/{total_chapters} 章节已保存", flush=True)
    
    print(f"\n  所有 {total_chapters} 章节翻译完成！", flush=True)

def generate_output_filename(original_filename: str, target_lang: str, model_info: dict, style: str) -> str:
    from datetime import datetime
    
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
    
    has_inprogress, inprogress_path, progress_file = check_inprogress(args.input_file)
    progress = ProgressState(args.input_file)
    
    if has_inprogress:
        print("\n  发现未完成的翻译任务，自动继续...")
        progress.load()
    
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
        working_file = inprogress_path if has_inprogress else args.input_file
        chapters = parse_ebook(working_file)
    except Exception as e:
        print(f"  [Error] Failed to parse ebook: {e}")
        sys.exit(1)
    
    total_text_nodes = sum(len(collect_text_nodes(c['soup'])) for c in chapters)
    if has_inprogress:
        progress.data["total_chapters"] = len(chapters)
        progress.save()
        processed = len(progress.data.get("processed_chapters", []))
        print(f"  Found {len(chapters)} chapters, {total_text_nodes} text nodes")
        print(f"  {processed} 章节已翻译，将跳过")
    else:
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
    
    if has_inprogress:
        output_path = progress.data.get("output_file", "")
        inprogress_path = progress.data.get("inprogress_file", "")
        print(f"  Output:   {output_path} (inprogress)")
    else:
        if args.output is None:
            print("  Generating filename...")
            original_name = os.path.splitext(os.path.basename(args.input_file))[0]
            args.output = generate_output_filename(original_name, args.target, model_info, args.prompt_style)
        
        output_path = args.output
        print(f"  Output:   {output_path}")
        
        inprogress_path = os.path.splitext(args.input_file)[0] + '.inprogress.epub'
        print(f"  创建 inprogress 文件: {inprogress_path}")
        
        progress.set_output(output_path, inprogress_path, args.prompt_style, args.target)
    
    exporter = EpubExporter()
    if not has_inprogress:
        exporter.export(chapters, inprogress_path, args.input_file)
    
    print("\n[3/4] Translating chapters...")
    translate_chapters(chapters, args.source, args.target, translator, max_para_tokens, progress, exporter, args.input_file)
    
    print("\n[4/4] Finalizing...")
    try:
        final_output = progress.data.get("output_file", args.output)
        if final_output != inprogress_path:
            save_translated_ebook(chapters, final_output, args.input_file)
            print(f"  删除 inprogress 文件: {inprogress_path}", flush=True)
            os.remove(inprogress_path)
        else:
            print(f"  最终文件: {final_output}")
        print(f"  删除进度文件: {progress.progress_file}", flush=True)
        progress.delete()
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
