from typing import List, Dict, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD  = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image: Image, 
           size: Tuple[int, int], 
           resample: Image.Resampling = None, 
           reducing_gap: Optional[int] = None
           ) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
        image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std:  Union[float, Iterable[float]],
        ) -> np.ndarray:
    
    mean = np.array(mean, dtype=image.dtype)
    std  = np.array(std, dtype=image.dtype)

    image = (image - mean) / std
    return image


def process_images(
        images: List[Image.Image],
        size:   Dict[str, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std:  Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    
    height, width = size[0], size[1]
    
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]

    images = [np.array(image) for image in images]
    
    images = [rescale(image, rescale_factor=rescale_factor) for image in images]
    
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    images = [image.transpose(2, 0, 1) for image in images]

    return images


class PaliGemmaProcessor:

    # Placeholder token for the image
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        # Sequence length of the image tokens
        self.image_seq_length = num_image_tokens

        # Size of the image
        self.image_size = image_size

        # Add the image token to the special tokens
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # Create extra tokens for localization and segmentation. In our particular case, we are not going to use them.
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]

        # Add the extra tokens to the tokenizer
        tokenizer.add_tokens(EXTRA_TOKENS)

        # Get the image token id
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # Disable the bos and eos tokens
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
            self,
            text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = True,
    ) -> dict:
        # From a list of images and prompts, we process the images by resizing, rescaling, normalizing and converting them to a tensor.
        # We also create the input prompt by adding how many image tokens we have in the sequence length, the bos token and the prefix prompt, 
        # adding at the end "\n" which is the SEQ token.
        # Finally, we tokenize the input strings and add the pixel values to the inputs.

        # Check if the number of images and prompts are both 1
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images=images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )

        # Stack the pixel values along the first axis. This is necessary because the model expects a batch of images.
        pixel_values = np.stack(pixel_values, axis=0)

        # Convert the pixel values to a tensor
        pixel_values = torch.tensor(pixel_values)

        # Create the input strings
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        # Tokenize the input strings
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )

        # Add the pixel values to the inputs
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
