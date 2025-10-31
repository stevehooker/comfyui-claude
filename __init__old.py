from .nodes.nodes import CombineTexts, DescribeImage, TransformText
from .nodes.advanced_nodes import (
    PromptEngineer,
    ContextAwareDescribe,
    IterativeRefine,
    PromptChain
)

NODE_CLASS_MAPPINGS = {
    # Original nodes
    'Describe Image': DescribeImage,
    'Combine Texts': CombineTexts,
    'Transform Text': TransformText,
    # Advanced nodes
    'Claude Prompt Engineer': PromptEngineer,
    'Claude Context Aware Describe': ContextAwareDescribe,
    'Claude Iterative Refine': IterativeRefine,
    'Claude Prompt Chain': PromptChain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Original nodes
    'Describe Image': 'Describe Image',
    'Combine Texts': 'Combine Texts',
    'Transform Text': 'Transform Text',
    # Advanced nodes with clear display names
    'Claude Prompt Engineer': 'Prompt Engineer (Claude)',
    'Claude Context Aware Describe': 'Context-Aware Describe (Claude)',
    'Claude Iterative Refine': 'Iterative Refine (Claude)',
    'Claude Prompt Chain': 'Prompt Chain (Claude)',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
