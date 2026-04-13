# translate_book.py 使用说明

## 1. 环境要求

- Python 3.8 及以上
- LM Studio 本地服务
- 已载入可用模型（例如 `qwen2.5-14b-instruct`、`llama2` 等）

## 2. 依赖安装（必备）

```bash
pip install requests beautifulsoup4 lxml
```

（脚本会优先尝试 `lxml-xml`，回退到 `html.parser`；`requests` 是 API 请求必备）

## 3. 运行前准备

1. 启动 LM Studio
2. 在 `Developer` 中点击 `Start Server`
3. 确保模型页面加载了模型

### 推荐模型设置（LM Studio 内）
- Context Length: 与模型最大值匹配，可在 4096/8192/16384 之间
- Flash Attention：开启（如果可用）
- Offload KV Cache to GPU：开启（如可用）

## 4. 脚本功能概览

`translate_book.py` 主要功能：

- 将 EPUB（XHTML）逐页逐段翻译为中英文双语
- 保留原结构、CSS、图片、交叉引用等 epub 元数据
- 动态基于模型 context length 计算 `MAX_TOKENS` 与 `CHUNK_SIZE`
- 支持批量翻译（`batch_translate_texts`）+ 并行回退
- 检测“诗歌/短文本/长文本”切换最优策略
- 支持自动续译（progress 文件）- 实时显示文件处理进度和章节内翻译进度
- 增强的超时处理（读取超时10分钟）
## 5. 关键策略

### 5.1 动态 `CHUNK_SIZE` 和 `MAX_TOKENS`
`check_model_loaded()` 中：

- 从 LM Studio API 获取模型详细信息（`/api/v1/models` 或 `/v1/models`）
- **优先获取运行时上下文长度**：`loaded_instances[0].config.context_length`（用户实际设置的值）
- 回退到模型最大能力：`max_context_length`
- `MODEL_CONTEXT_LENGTH` 根据模型名称推断（Qwen默认16384，其他8192）
- `MAX_TOKENS = min(16384, max(1024, MODEL_CONTEXT_LENGTH - 512))`
- `CHUNK_SIZE = min(8000, max(1000, int((MODEL_CONTEXT_LENGTH - 512) * 4 * 0.8)))`

### 5.2 模型元数据获取
- 自动从 LM Studio API 获取：发布者、显示名称、量化名称
- 支持手动指定：`--model-meta "作者,模型名,量化名"`
- 用于生成输出文件名：`{原文件名}_{发布者}_{模型名}_{量化名}_{时间戳}.epub`

### 5.3 内容分析自动调优（`analyze_epub_paragraphs`）

- 统计块级段落 (`p/h* /li/div/blockquote`) 长度
- 计算平均段长、推荐分块：`min(math.ceil(avg_para_length * 8), CHUNK_SIZE)`

### 5.4 帧句式保护（超长段按标点切分）

内置：`split_text_by_punctuation(text, max_chunk)`

- 以中文/英文句末符（`。！？；;.!?`）拆句
- 尽量组合到不超 `max_chunk`
- 超长句直接按字符拆

在 `translate_long_text` 里：

- 先尝试完整段落翻译
- 若段落超过 `chunk_size`，按空行分段
- 段落仍过长，按标点拆分（`split_text_by_punctuation`）
- 再失败才按字符切分

### 5.5 诗歌模式（`make_bilingual_soup`）

- 平均行长 < 60 → 认为诗歌，进行整体翻译（`translate_text(full_poetry_text)`）
- 结果行数不足则回退批量翻译

### 5.6 批处理翻译

`batch_translate_texts(texts)`：

- **动态批次大小**：根据模型上下文长度和文本预估token数，动态计算每批最多包含的段落数（最多12段）
- **格式错误重试**：若批处理输出格式不正确，会降低温度重试（最多2次）
- 少量(<=8)文本段落可一次提交，结合 `===SEGMENT===` 分隔
- 优先解析 JSON/文本切片，失败回退并行 `translate_text`（线程池，默认 2 个并发）
- 支持递归分批处理大量文本

### 5.7 并发翻译

- 使用 `concurrent.futures.ThreadPoolExecutor` 进行并行翻译
- 默认并发数：2，可通过 `--max-workers` 调整
- 适用于本地模型，避免阻塞
- 实时显示章节内翻译进度（`章节内进度: X/Y 行`）

## 6. 使用范例

```bash
python translate_book.py 'The Wood (H. P. Lovecraft).epub' --output './TheWood.bilingual.epub' --profile 1
```

可选参数：

- `--profile 0|1|2`：技术翻译/文学翻译/社科非虚构翻译（必选）
- `--output <target.epub>`：输出路径（默认自动生成带时间戳的文件名）
- `--context-length <tokens>`：强制上下文长度，覆盖模型推断
- `--model-meta "作者,模型名,量化名"`：手动指定模型元数据
- `--resume` 或 `--continue`：自动续译模式，不询问直接继续上次翻译
- `--force`：忽略进度文件，强制重新开始
- `--no-batch`：禁用批处理（逐行翻译）
- `--max-batch-lines <N>`：最大批处理行数，默认自动计算
- `--max-workers <N>`：并发翻译线程数，默认 2

## 7. 输出文件名自动生成

若不指定 `--output`，自动生成格式：
```
{原文件名}_{发布者}_{模型名}_{量化名}_{YYYYMMDD_HHMMSS}.epub
```

例如：
```
The Wood (H. P. Lovecraft)_bartowski_Qwen2.5 14B Instruct_Q6_K_L_20260409_125149.epub
```

## 8. 进度与错误恢复

- 生成：`<output>.progress.json` 和 `<output>.inprogress.epub`
- 出错可保留记录，下次同目标带 `--resume` 续译
- 支持检测多个未完成进度，选择恢复或重新开始

## 9. 性能提示

- LM Studio 与模型本身为核心瓶颈，建议运行于独立机器
- 自定义 `MAX_WHOLE_ATTEMPT_LIMIT`、`BATCH_ENABLED` 可调脚本顶部变量
- 诗歌、短文本会优先简化 API 调用，长段下按分析结果分块
- **批处理参数动态计算**：`max_batch_lines = max(3, int(MODEL_CONTEXT_LENGTH / 30 / 4))`
- **超时设置**：连接超时60秒，读取超时600秒（10分钟）

## 10. 常见问题

- `MAX_TOKENS 未初始化`：请先启动 LM Studio 并加载模型。
- `No models loaded`：模型未加载/路径错误，检查 LM Studio Model 页。
- 400 及 `Model reloaded`：可能超了输入限制，降低上下文长度或分块大小

欢迎根据文档改造出自己的分段优先级、标点策略、并发参数。

安装后，程序将使用 `lxml-xml` 解析 EPUB 中的 XHTML 文件，能更好地保留命名空间和 XML 结构。