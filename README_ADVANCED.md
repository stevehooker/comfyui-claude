# ComfyUI Claude Advanced

Enhanced ComfyUI nodes for Claude integration with advanced prompt engineering and context-aware features.

## Features

### Original Nodes
- **Describe Image**: Basic image description using Claude's vision API
- **Combine Texts**: Combine two text inputs using Claude
- **Transform Text**: Transform text using Claude

### New Advanced Nodes

#### 🎯 Prompt Engineer (Claude)
Advanced prompt engineering with templates and structured approaches:
- **Template Styles**: Qwen-style, Chain of Thought, Structured Analysis, Creative Narrative, Technical Documentation
- **Output Formats**: Plain text, Markdown, JSON, XML
- **Detail Levels**: Concise, Standard, Detailed, Exhaustive
- **Few-shot Learning**: Add example input/output pairs
- **Creativity Control**: Adjust response creativity (0.0-1.0)

#### 🖼️ Context-Aware Describe (Claude)
Multi-image analysis with relationship understanding:
- **Comparison Modes**:
  - Individual: Describe each image separately
  - Comparative: Focus on differences and similarities
  - Sequential: Treat as a sequence or progression
  - Holistic: Describe as parts of a whole
- **Context Preservation**: Use previous descriptions for continuity
- **Focus Elements**: Customizable analysis focus areas

#### ♻️ Iterative Refine (Claude)
Qwen-inspired iterative improvement system:
- **Refinement Strategies**:
  - Clarify: Improve clarity and understanding
  - Expand: Add relevant details and examples
  - Focus: Make more concise while keeping key info
  - Restructure: Improve organization and flow
  - Enhance Quality: Overall quality improvement
- **Multiple Iterations**: Up to 3 refinement passes
- **Structure Preservation**: Option to maintain original structure
- **Accumulative Improvements**: Build upon previous iterations

#### ⛓️ Prompt Chain (Claude)
Chain multiple prompts for complex multi-step processing:
- **3-Step Processing**: Chain up to 3 sequential prompts
- **Flexible Input**: Use text, images, or both
- **Context Passing**: Pass results between steps
- **Combination Modes**:
  - Append: Simple concatenation
  - Synthesize: AI-powered synthesis
  - Extract Key Points: Distill main insights
  - Structured Merge: Formatted combination

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stevehooker/comfyui-claude.git
```

2. Install requirements:
```bash
pip install -r comfyui-claude/requirements.txt
```

3. Restart ComfyUI

## Usage Examples

### Example 1: Advanced Prompt Engineering
1. Add "Prompt Engineer (Claude)" node
2. Set template_style to "qwen_style" for Qwen-inspired prompting
3. Adjust detail_level and output_format as needed
4. Connect to your image generation pipeline

### Example 2: Multi-Image Context Analysis
1. Add "Context-Aware Describe (Claude)" node
2. Connect multiple images
3. Set comparison_mode to "comparative" or "sequential"
4. Use the description for informed image generation

### Example 3: Iterative Refinement
1. Generate initial description with any node
2. Add "Iterative Refine (Claude)" node
3. Set refinement_strategy and iteration_count
4. Get progressively improved results

### Example 4: Complex Analysis Chain
1. Add "Prompt Chain (Claude)" node
2. Define step-by-step analysis prompts
3. Enable context passing between steps
4. Get comprehensive, structured analysis

## Models Supported

- Claude Opus 4.1 (claude-opus-4-1-20250805)
- Claude Sonnet 4 (claude-sonnet-4-20250514)
- Claude 3.5 Haiku (Latest & specific versions)
- Claude 3.5 Sonnet (Latest & specific versions)
- Claude 3 Opus (Latest & specific versions)
- Legacy models for compatibility

## Error Handling

Enhanced error handling provides clear feedback for:
- Rate limit issues
- Authentication problems
- Image format issues
- API connection problems

## Credits

Original nodes by tkreuziger and harelc
Advanced features by stevehooker

## License

Same as original repository
