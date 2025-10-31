"""Specialized node for generating Qwen edit prompts."""

from comfyui_types import (
    BooleanInput,
    ChoiceInput,
    ComfyUINode,
    ImageInput,
    IntegerInput,
    StringInput,
    StringOutput,
)

from .ai import describe_image, models, run_prompt


class QwenPromptGenerator(ComfyUINode):
    """Generate action-oriented prompts specifically for Qwen image editing."""
    
    category = 'Claude/Qwen'
    
    # Problem description
    problem_description = StringInput(
        multiline=True,
        default="Foot appears to float above ground",
        display_name="What needs fixing?"
    )
    
    # Edit targets
    edit_ground = BooleanInput(default=True, display_name="Fix ground contact")
    edit_shadows = BooleanInput(default=True, display_name="Fix shadows/lighting")
    edit_integration = BooleanInput(default=True, display_name="Fix integration/blending")
    
    # Specific parameters
    light_direction = ChoiceInput(
        choices=['top-left', 'top-right', 'middle-left', 'middle-right', 
                'bottom-left', 'bottom-right', 'ambient'],
        default='middle-right',
        display_name="Light source direction"
    )
    
    light_quality = ChoiceInput(
        choices=['harsh', 'soft', 'golden hour', 'overcast', 'dramatic'],
        default='golden hour',
        display_name="Light quality"
    )
    
    ground_type = ChoiceInput(
        choices=['rocky', 'soil', 'grass', 'sand', 'stone', 'mixed'],
        default='rocky',
        display_name="Ground type"
    )
    
    # Output style
    output_style = ChoiceInput(
        choices=['single_line', 'bullet_points', 'numbered_steps'],
        default='single_line',
        display_name="Output format"
    )
    
    # Advanced
    max_words = IntegerInput(default=50, min=20, max=100, display_name="Max words")
    
    model = ChoiceInput(choices=models)
    api_key = StringInput()
    
    qwen_prompt = StringOutput()
    
    def execute(
        self,
        problem_description: str,
        edit_ground: bool,
        edit_shadows: bool,
        edit_integration: bool,
        light_direction: str,
        light_quality: str,
        ground_type: str,
        output_style: str,
        max_words: int,
        model: str,
        api_key: str,
    ) -> tuple[str]:
        """Generate a Qwen-optimized edit prompt."""
        
        # Build the command components
        commands = []
        
        if edit_ground:
            commands.append(f"Anchor firmly onto {ground_type} ground")
            commands.append(f"Create realistic deformation in {ground_type}")
            commands.append("Compress terrain at weight points")
        
        if edit_shadows:
            light_dir_text = light_direction.replace('-', ' ')
            commands.append(f"Cast {light_quality} shadows from {light_dir_text}")
            if edit_ground:
                commands.append("Add contact shadows at ground meeting points")
        
        if edit_integration:
            commands.append("Blend naturally with surface texture")
            commands.append("Match environmental lighting")
        
        # System prompt for Claude to convert to Qwen format
        system_prompt = f"""You are a Qwen edit prompt optimizer. 
Convert the given commands into a {max_words}-word maximum edit instruction.
Use ONLY action verbs. No descriptions or explanations.
Output format: {output_style}"""
        
        # Construct the prompt
        base_prompt = f"""Problem: {problem_description}

Required edits:
{chr(10).join(commands)}

Combine these into a {max_words}-word Qwen edit instruction using only action verbs."""
        
        # Get the refined prompt
        result = run_prompt(base_prompt, system_prompt, model, api_key)
        
        # Format based on style
        if output_style == 'single_line':
            result = result.replace('\n', ', ').replace('- ', '').strip()
        
        return (result,)


class QwenFromImage(ComfyUINode):
    """Analyze an image and generate Qwen edit commands directly."""
    
    category = 'Claude/Qwen'
    
    image = ImageInput()
    
    # What to analyze
    analyze_ground = BooleanInput(default=True, display_name="Analyze ground issues")
    analyze_lighting = BooleanInput(default=True, display_name="Analyze lighting issues")
    analyze_integration = BooleanInput(default=True, display_name="Analyze integration")
    
    # Known issues (optional)
    known_issues = StringInput(
        multiline=True,
        required=False,
        default="",
        display_name="Known issues (optional)"
    )
    
    # Target state
    target_description = StringInput(
        multiline=False,
        default="Realistic ground contact with proper shadows",
        display_name="Desired result"
    )
    
    model = ChoiceInput(choices=models)
    api_key = StringInput()
    
    qwen_prompt = StringOutput()
    analysis = StringOutput()
    
    def execute(
        self,
        image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
        analyze_ground: bool,
        analyze_lighting: bool,
        analyze_integration: bool,
        known_issues: str,
        target_description: str,
        model: str,
        api_key: str,
    ) -> tuple[str, str]:
        """Analyze image and generate Qwen commands."""
        
        # Build analysis focus
        focus_areas = []
        if analyze_ground:
            focus_areas.append("ground contact and deformation")
        if analyze_lighting:
            focus_areas.append("shadows and lighting consistency")
        if analyze_integration:
            focus_areas.append("edge blending and integration")
        
        # System prompt for analysis
        analysis_system = f"""You are a visual effects specialist identifying specific fixes needed.
Focus on: {', '.join(focus_areas)}
Output format: List specific problems that need editing, not descriptions."""
        
        analysis_prompt = f"""Identify editing requirements to achieve: {target_description}
{f'Known issues: {known_issues}' if known_issues else ''}
List ONLY what needs to be changed, as brief statements."""
        
        # Get analysis
        analysis = describe_image(
            image,
            analysis_prompt,
            analysis_system,
            model,
            api_key
        )
        
        # Convert to Qwen commands
        command_system = """Convert problem statements to Qwen edit commands.
Use pattern: [VERB] [OBJECT] [SPECIFICATION]
Examples: 'Anchor foot to ground', 'Create shadow from right', 'Deepen contact depression'
Maximum 50 words total. Single line output."""
        
        command_prompt = f"""Convert these issues to edit commands:
{analysis}

Target result: {target_description}"""
        
        qwen_prompt = run_prompt(
            command_prompt,
            command_system,
            model,
            api_key
        )
        
        # Clean up the output
        qwen_prompt = qwen_prompt.replace('\n', ', ').replace('- ', '').strip()
        
        return (qwen_prompt, analysis)
