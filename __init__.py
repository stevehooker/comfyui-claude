from .nodes.nodes import CombineTexts, DescribeImage, DescribeImageCached, TransformText

NODE_CLASS_MAPPINGS = {
    'Describe Image': DescribeImage,
    'Describe Image (Cached)': DescribeImageCached,
    'Combine Texts': CombineTexts, 
    'Transform Text': TransformText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Describe Image': 'Describe Image',
    'Describe Image (Cached)': 'Describe Image (Cached)',
    'Combine Texts': 'Combine Texts', 
    'Transform Text': 'Transform Text',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
