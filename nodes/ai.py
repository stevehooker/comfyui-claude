"""AI functions."""

import base64
import io
import logging

import anthropic
from PIL import Image
import numpy as np

models = [
    # Latest versions (recommended)
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-opus-latest',
    # Specific versions - newest
    'claude-opus-4-1-20250805',
    'claude-sonnet-4-20250514',
    'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-20241022',
    # Older versions for compatibility
    'claude-3-opus-20240229',
    'claude-3-haiku-20240307',
    'claude-3-sonnet-20240229',
]


def run_prompt(
    prompt: str, system_prompt: str, model: str, api_key: str
) -> str:
    """Execute a simple prompt with Claude.

    Args:
        prompt (str): The prompt to execute.
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except anthropic.RateLimitError as e:
        error_msg = f"Rate limit exceeded: {str(e)}. Please wait and try again."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except anthropic.AuthenticationError as e:
        error_msg = f"Authentication failed: {str(e)}. Please check your API key."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except Exception as e:
        error_msg = f'Unexpected error in run_prompt: {str(e)}'
        logging.exception(error_msg)
        return f"ERROR: {error_msg}"

    return ''


def describe_image(
    image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Send an image to Claude's vision API.

    Args:
        image (torch.Tensor): The image to describe.
        prompt (str): The prompt to use.
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        api_key (str): The API key to use.

    Returns:
        str: The result of the prompt.
    """
    try:
        # Validate inputs
        if image is None:
            error_msg = "No image provided to describe_image function"
            logging.error(error_msg)
            return f"ERROR: {error_msg}"
        
        if not prompt:
            prompt = "Describe this image in detail."
            
        # Log image info for debugging
        logging.info(f"Processing image with shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        logging.info(f"Image dtype: {image.dtype if hasattr(image, 'dtype') else 'unknown'}")
        
        # Convert tensor to image with better error handling
        try:
            # Ensure image is in the right format
            if len(image.shape) == 4:
                image = image.squeeze(0)  # Remove batch dimension
            
            # Handle different tensor ranges
            if image.max() <= 1.0:
                image_tensor = image * 255
            else:
                image_tensor = image
                
            # Convert to numpy array
            image_array = image_tensor.byte().cpu().numpy()
            
            # Handle different channel arrangements
            if len(image_array.shape) == 3:
                if image_array.shape[0] == 3:  # CHW format
                    image_array = np.transpose(image_array, (1, 2, 0))
                # If shape[2] == 3, it's already HWC format
            
            # Convert to PIL Image
            if image_array.shape[-1] == 4:  # RGBA
                pil_image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
            else:
                pil_image = Image.fromarray(image_array, mode='RGB')
                
        except Exception as e:
            error_msg = f"Failed to convert tensor to image: {str(e)}"
            logging.error(error_msg)
            logging.error(f"Image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
            return f"ERROR: {error_msg}"
        
        # Save image to bytes
        buffered = io.BytesIO()
        pil_image.save(buffered, format='JPEG', quality=95)
        img_data = buffered.getvalue()
        
        # Check image size
        img_size_mb = len(img_data) / (1024 * 1024)
        if img_size_mb > 10:  # Claude has a 10MB limit
            logging.warning(f"Image size ({img_size_mb:.2f}MB) may be too large. Compressing...")
            buffered = io.BytesIO()
            pil_image.save(buffered, format='JPEG', quality=70)
            img_data = buffered.getvalue()

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': base64.b64encode(img_data).decode(
                                    'utf-8'
                                ),
                            },
                        },
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                    ],
                }
            ],
        )

        if message and message.content and len(message.content) > 0:
            return message.content[0].text  # type: ignore  # noqa: PGH003

    except anthropic.RateLimitError as e:
        error_msg = f"Rate limit exceeded: {str(e)}. Please wait and try again."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except anthropic.AuthenticationError as e:
        error_msg = f"Authentication failed: {str(e)}. Please check your API key."
        logging.error(error_msg)
        return f"ERROR: {error_msg}"
    except Exception as e:
        error_msg = f'Unexpected error in describe_image: {str(e)}'
        logging.exception(error_msg)
        return f"ERROR: {error_msg}"

    return ''
