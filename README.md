# Ebook Translator

基于本地 LLM 的 EPUB 电子书翻译工具，将英文书籍翻译为双语对照版本。

## 功能特性

- 支持三种翻译专家模式：技术翻译、文学翻译、社科非虚构翻译
- 保留原书结构和样式，生成双语对照版本
- 智能批处理，提升翻译效率
- 支持断点续译
- 动态计算最优批次大小，充分利用模型上下文

## 环境要求

- Python 3.8+
- [LM Studio](https://lmstudio.ai/) 已加载模型

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python ebook-translator.py <input.epub> --profile <0|1|2>
```

### 翻译专家

| 编号 | 名称 | 适用场景 |
|------|------|----------|
| 0 | 技术翻译 | 技术书籍、编程文档 |
| 1 | 文学翻译 | 小说、文学作品 |
| 2 | 社科非虚构翻译 | 科普、社会科学、商业书籍 |

### 示例

```bash
# 技术书籍翻译
python ebook-translator.py book.epub --profile 0

# 文学翻译
python ebook-translator.py novel.epub --profile 1

# 续译未完成的翻译
python ebook-translator.py book.epub --resume
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--profile` | 选择翻译专家（0/1/2） |
| `--output` | 指定输出文件 |
| `--resume` | 自动续译上次未完成的翻译 |
| `--force` | 强制重新开始，忽略已有进度 |
| `--no-batch` | 禁用批处理模式 |
| `--max-workers` | 并发线程数（默认2） |
| `--context-length` | 手动指定模型上下文长度 |
| `--model-meta` | 手动指定模型元数据，格式：`作者,模型名,量化名` |

## 工作原理

1. 解压 EPUB 文件
2. 解析 XHTML 内容
3. 调用本地 LLM API 翻译（通过 LM Studio）
4. 在原文后插入翻译结果（带 `translation` class）
5. 重新打包为 EPUB

## 依赖

- requests
- beautifulsoup4
- lxml
- ebooklib
