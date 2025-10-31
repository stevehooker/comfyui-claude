"""Advanced nodes for Claude integration with prompt engineering and context awareness."""

from comfyui_types import (
    BooleanInput,
    ChoiceInput,
    ComfyUINode,
    FloatInput,
    ImageInput,
    IntegerInput,
    StringInput,
    StringOutput,
)

from .ai import describe_image, models, run_prompt
import json
import re


# Prompt engineering templates
PROMPT_TEMPLATES = {
    'qwen_style': """You are a helpful assistant. Please follow these guidelines:
1. Be precise and detailed in your analysis
2. Structure your response clearly
3. Use specific technical terminology when appropriate
4. Maintain objectivity while being comprehensive

Task: {task}
Context: {context}
Requirements: {requirements}""",
    
    'chain_of_thought': """Let's approach this step-by-step.

First, I need to understand: {task}
Context provided: {context}

Let me think through this systematically:
1. Initial observation
2. Detailed analysis
3. Key insights
4. Final conclusion

{requirements}""",
    
    'structured_analysis': """Analyze the following according to these dimensions:
- Visual/Content Overview
- Technical Details
- Artistic/Aesthetic Elements
- Contextual Significance
- Potential Applications

Task: {task}
Context: {context}
Additional Requirements: {requirements}""",
    
    'creative_narrative': """Create an engaging narrative that:
- Captures the essence of the subject
- Uses vivid, descriptive language
- Maintains coherence and flow
- Incorporates subtle details

Task: {task}
Context: {context}
Style Requirements: {requirements}""",
    
    'technical_documentation': """Generate technical documentation following these standards:
- Use precise terminology
- Include all relevant parameters
- Structure information hierarchically
- Provide clear explanations

Task: {task}
Technical Context: {context}
Documentation Requirements: {requirements}"""
}


class PromptEngineer(ComfyUINode):
    """Advanced prompt engineering with templates and structured approaches."""
    
    category = 'Claude/Advanced'
    
    # Basic inputs
    base_prompt = StringInput(multiline=True, default="Describe in detail")
    context = StringInput(multiline=True, required=False, default="")
    requirements = StringInput(multiline=True, required=False, default="")
    
    # Template selection
    template_style = ChoiceInput(
        choices=['none', 'qwen_style', 'chain_of_thought', 'structured_analysis', 
                'creative_narrative', 'technical_documentation'],
        default='none'
    )
    
    # Advanced options
    output_format = ChoiceInput(
        choices=['plain', 'markdown', 'json', 'xml'],
        default='plain'
    )
    
    detail_level = ChoiceInput(
        choices=['concise', 'standard', 'detailed', 'exhaustive'],
        default='standard'
    )
    
    # Few-shot examples
    example_input = StringInput(multiline=True, required=False, default="")
    example_output = StringInput(multiline=True, required=False, default="")
    
    # Temperature control hint
    creativity = FloatInput(default=0.7, min=0.0, max=1.0, step=0.1)
    
    # Model selection
    model = ChoiceInput(choices=models)
    api_key = StringInput()
    
    engineered_prompt = StringOutput()
    system_prompt_out = StringOutput()
    
    def execute(
        self,
        base_prompt: str,
        context: str,
        requirements: str,
        template_style: str,
        output_format: str,
        detail_level: str,
        example_input: str,
        example_output: str,
        creativity: float,
        model: str,
        api_key: str,
    ) -> tuple[str, str]:
        """Engineer an advanced prompt with templates and structure."""
        
        # Build system prompt based on settings
        system_parts = []
        
        if template_style != 'none' and template_style in PROMPT_TEMPLATES:
            system_parts.append(PROMPT_TEMPLATES[template_style].format(
                task=base_prompt,
                context=context,
                requirements=requirements
            ))
        
        # Add detail level instructions
        detail_instructions = {
            'concise': 'Be brief and to the point. Focus on key information only.',
            'standard': 'Provide a balanced level of detail.',
            'detailed': 'Include comprehensive details and thorough explanations.',
            'exhaustive': 'Provide exhaustive detail, leaving nothing unexamined.'
        }
        system_parts.append(detail_instructions[detail_level])
        
        # Add format instructions
        if output_format == 'markdown':
            system_parts.append("Format your response using proper Markdown syntax.")
        elif output_format == 'json':
            system_parts.append("Structure your response as valid JSON.")
        elif output_format == 'xml':
            system_parts.append("Structure your response using XML tags.")
        
        # Add creativity hint
        if creativity < 0.3:
            system_parts.append("Be factual and objective. Avoid speculation.")
        elif creativity > 0.7:
            system_parts.append("Feel free to be creative and explore interesting angles.")
        
        system_prompt = "\n\n".join(system_parts)
        
        # Build the main prompt
        prompt_parts = [base_prompt]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        if requirements:
            prompt_parts.append(f"Specific requirements: {requirements}")
        
        # Add few-shot example if provided
        if example_input and example_output:
            prompt_parts.append(f"\nExample:\nInput: {example_input}\nOutput: {example_output}\n\nNow, following this pattern:")
        
        engineered_prompt = "\n\n".join(prompt_parts)
        
        return (engineered_prompt, system_prompt)


class ContextAwareDescribe(ComfyUINode):
    """Describe images with awareness of multiple images and their relationships."""
    
    category = 'Claude/Advanced'
    
    # Multiple image inputs (ComfyUI will handle as batch)
    images = ImageInput()
    
    # Context settings
    comparison_mode = ChoiceInput(
        choices=['individual', 'comparative', 'sequential', 'holistic'],
        default='individual'
    )
    
    # Additional context
    previous_description = StringInput(multiline=True, required=False, default="")
    scene_context = StringInput(multiline=True, required=False, default="")
    
    # Focus areas
    focus_elements = StringInput(
        default="subjects, colors, composition, mood, technical quality",
        multiline=False
    )
    
    # Prompt customization
    base_prompt = StringInput(
        multiline=True,
        default="Analyze these images with attention to their relationships and context."
    )
    
    model = ChoiceInput(choices=models)
    api_key = StringInput()
    
    description = StringOutput()
    
    def execute(
        self,
        images: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
        comparison_mode: str,
        previous_description: str,
        scene_context: str,
        focus_elements: str,
        base_prompt: str,
        model: str,
        api_key: str,
    ) -> tuple[str]:
        """Describe images with contextual awareness."""
        
        # Build context-aware system prompt
        system_prompt = f"""You are an expert image analyst with deep understanding of visual context and relationships.
        
Focus on these elements: {focus_elements}

Analysis mode: {comparison_mode}
- individual: Describe each image separately but note connections
- comparative: Focus on differences and similarities
- sequential: Treat as a sequence or progression
- holistic: Describe as parts of a whole scene or concept
"""
        
        if scene_context:
            system_prompt += f"\n\nScene context: {scene_context}"
        
        if previous_description:
            system_prompt += f"\n\nPrevious analysis for context: {previous_description}"
        
        # Build the prompt
        full_prompt = base_prompt
        
        if comparison_mode == 'comparative':
            full_prompt += "\n\nCompare and contrast these images, highlighting key differences and similarities."
        elif comparison_mode == 'sequential':
            full_prompt += "\n\nDescribe these images as a sequence, noting progressions and changes."
        elif comparison_mode == 'holistic':
            full_prompt += "\n\nDescribe how these images relate to form a complete picture or narrative."
        
        # Process images (handling batch if multiple)
        description = describe_image(images, full_prompt, system_prompt, model, api_key)
        
        return (description,)


class IterativeRefine(ComfyUINode):
    """Iteratively refine prompts for better results, inspired by Qwen-style prompting."""
    
    category = 'Claude/Advanced'
    
    initial_result = StringInput(multiline=True)
    
    refinement_strategy = ChoiceInput(
        choices=['clarify', 'expand', 'focus', 'restructure', 'enhance_quality'],
        default='enhance_quality'
    )
    
    refinement_instructions = StringInput(
        multiline=True,
        default="Improve this description by adding more specific details and technical accuracy."
    )
    
    # Qwen-style iteration controls
    iteration_count = IntegerInput(default=1, min=1, max=3)
    preserve_structure = BooleanInput(default=True)
    accumulate_improvements = BooleanInput(default=True)
    
    # Quality metrics to optimize
    optimize_for = StringInput(
        default="clarity, detail, accuracy, readability",
        multiline=False
    )
    
    model = ChoiceInput(choices=models)
    api_key = StringInput()
    
    refined_result = StringOutput()
    improvement_notes = StringOutput()
    
    def execute(
        self,
        initial_result: str,
        refinement_strategy: str,
        refinement_instructions: str,
        iteration_count: int,
        preserve_structure: bool,
        accumulate_improvements: bool,
        optimize_for: str,
        model: str,
        api_key: str,
    ) -> tuple[str, str]:
        """Iteratively refine text using Qwen-style approach."""
        
        strategies = {
            'clarify': "Make this clearer and more understandable while maintaining accuracy:",
            'expand': "Expand this with more relevant details and examples:",
            'focus': "Make this more focused and concise while keeping key information:",
            'restructure': "Reorganize this for better flow and logical structure:",
            'enhance_quality': "Enhance the overall quality considering: "
        }
        
        system_prompt = f"""You are a meticulous editor focused on iterative improvement.
Optimization targets: {optimize_for}
{"Preserve the original structure and format." if preserve_structure else "Feel free to restructure as needed."}
{"Build upon previous improvements." if accumulate_improvements else "Fresh perspective each iteration."}
"""
        
        current_result = initial_result
        improvements = []
        
        for i in range(iteration_count):
            iteration_prompt = f"""Iteration {i+1} of {iteration_count}

{strategies[refinement_strategy]}{optimize_for if refinement_strategy == 'enhance_quality' else ''}

Current version:
{current_result}

Specific instructions: {refinement_instructions}

Provide the improved version:"""
            
            current_result = run_prompt(
                iteration_prompt,
                system_prompt,
                model,
                api_key
            )
            
            improvements.append(f"Iteration {i+1}: Applied {refinement_strategy} strategy")
        
        improvement_notes = "\n".join(improvements)
        
        return (current_result, improvement_notes)


class PromptChain(ComfyUINode):
    """Chain multiple prompts together for complex multi-step processing."""
    
    category = 'Claude/Advanced'
    
    # Input for chaining
    input_text = StringInput(multiline=True, required=False, default="")
    input_image = ImageInput(required=False)
    
    # Chain steps (up to 3)
    step1_prompt = StringInput(multiline=True, default="First, analyze the main subjects")
    step1_use_image = BooleanInput(default=True)
    
    step2_prompt = StringInput(multiline=True, required=False, default="Next, describe the context and environment")
    step2_use_previous = BooleanInput(default=True)
    
    step3_prompt = StringInput(multiline=True, required=False, default="Finally, synthesize insights")
    step3_use_all = BooleanInput(default=True)
    
    # Combination strategy
    combination_mode = ChoiceInput(
        choices=['append', 'synthesize', 'extract_key_points', 'structured_merge'],
        default='synthesize'
    )
    
    model = ChoiceInput(choices=models)
    api_key = StringInput()
    
    final_output = StringOutput()
    intermediate_results = StringOutput()
    
    def execute(
        self,
        input_text: str,
        input_image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
        step1_prompt: str,
        step1_use_image: bool,
        step2_prompt: str,
        step2_use_previous: bool,
        step3_prompt: str,
        step3_use_all: bool,
        combination_mode: str,
        model: str,
        api_key: str,
    ) -> tuple[str, str]:
        """Execute a chain of prompts with intelligent combination."""
        
        results = []
        context = input_text
        
        # Step 1
        if step1_use_image and input_image is not None:
            result1 = describe_image(
                input_image,
                step1_prompt + (f"\nContext: {context}" if context else ""),
                "You are performing step 1 of a multi-step analysis.",
                model,
                api_key
            )
        else:
            result1 = run_prompt(
                step1_prompt + (f"\nInput: {context}" if context else ""),
                "You are performing step 1 of a multi-step analysis.",
                model,
                api_key
            )
        results.append(f"Step 1: {result1}")
        
        # Step 2
        if step2_prompt:
            context2 = result1 if step2_use_previous else input_text
            result2 = run_prompt(
                step2_prompt + f"\nPrevious analysis: {context2}",
                "You are performing step 2 of a multi-step analysis.",
                model,
                api_key
            )
            results.append(f"Step 2: {result2}")
        else:
            result2 = ""
        
        # Step 3
        if step3_prompt and step3_use_all:
            all_context = f"Initial input: {input_text}\nStep 1 result: {result1}\nStep 2 result: {result2}"
            result3 = run_prompt(
                step3_prompt + f"\nAll previous analysis: {all_context}",
                "You are performing the final synthesis step.",
                model,
                api_key
            )
            results.append(f"Step 3: {result3}")
        else:
            result3 = result2 if result2 else result1
        
        # Combine results based on mode
        if combination_mode == 'append':
            final = "\n\n".join(results)
        elif combination_mode == 'synthesize':
            synthesis_prompt = f"Synthesize these analysis steps into a coherent whole:\n{chr(10).join(results)}"
            final = run_prompt(synthesis_prompt, "Create a unified analysis from multiple perspectives.", model, api_key)
        elif combination_mode == 'extract_key_points':
            extract_prompt = f"Extract the key points from this analysis:\n{chr(10).join(results)}"
            final = run_prompt(extract_prompt, "Extract and organize the most important insights.", model, api_key)
        else:  # structured_merge
            final = f"# Analysis Results\n\n" + "\n\n".join([f"## {r}" for r in results])
        
        intermediate = "\n---\n".join(results)
        
        return (final, intermediate)
