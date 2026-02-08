# ComfyUI and Claude
A set of custom nodes that are using Anthropic's Claude models for describing images and transforming texts.

## Fork Information

This is a fork of [harelc/comfyui-claude](https://github.com/harelc/comfyui-claude), which itself is based on the original work by [tkreuziger](https://github.com/tkreuziger/comfyui-claude). 

### Changes in This Fork

- **Updated Claude models**: Added support for Claude Opus 4.6, Sonnet 4.5, and Haiku 4.5 (February 2026)
- **Prompt caching support**: New DescribeImage (Cached) node for cost-effective batch image captioning
- **Improved error handling**: Better error messages and authentication feedback

## Setup

Clone this repository into your ComfyUI custom nodes folder and install the requirements:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stevehooker/comfyui-claude.git
cd comfyui-claude
pip install -r requirements.txt
```

Then restart ComfyUI.

Alternatively, if you're using ComfyUI Manager, you can install the original version from the package registry and then manually update to this fork.

## Requirements

You need an Anthropic API key that you must fill in the nodes. Learn more about
this [here](https://docs.anthropic.com/en/api/getting-started).

## Included nodes

1. **DescribeImage**: Takes an image as input and returns a textual description of it.
2. **DescribeImage (Cached)**: Optimised version for batch image captioning that uses prompt caching to reduce costs by up to 90% when processing multiple images with the same system prompt. Ideal for generating captions for LoRA training datasets.
3. **CombineTexts**: Combine two texts into something new with the help of Claude.
4. **TransformText**: Transforms an input text into some other text, ideal for rephrasing prompts or similar.

## Supported Models

This fork includes support for the latest Claude models (updated February 2026):

**Latest flagship models**:
- `claude-opus-4-6` - Most capable model for complex tasks
- `claude-sonnet-4-5-20250929` - Best balance of intelligence and speed
- `claude-haiku-4-5-20251001` - Fast and cost-effective

**Previous generation (still supported)**:
- `claude-sonnet-4-20250514`
- `claude-3-opus-20240229`
- `claude-3-5-haiku-20241022`

## Prompt Caching

The **DescribeImage (Cached)** node implements [Anthropic's prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) feature, which caches your system prompt across multiple API calls. This is particularly useful when:

- Processing batches of images with consistent captioning instructions
- Generating training data for LoRA models
- Running repetitive image analysis tasks

**Cost savings**: After the first image, cached system prompts cost only 10% of regular input tokens, resulting in significant savings for batch operations.

## Credits

- Original work by [tkreuziger](https://github.com/tkreuziger/comfyui-claude)
- Forked from [harelc](https://github.com/harelc/comfyui-claude)
- Extended by [stevehooker](https://github.com/stevehooker/comfyui-claude)
