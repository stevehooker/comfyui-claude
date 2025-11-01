from .nodes.nodes import CombineTexts, DescribeImage, TransformText

NODE_CLASS_MAPPINGS = {
    'Describe Image': DescribeImage,
    'Combine Texts': CombineTexts,
    'Transform Text': TransformText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Describe Image': 'Describe Image',
    'Combine Texts': 'Combine Texts', 
    'Transform Text': 'Transform Text',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
