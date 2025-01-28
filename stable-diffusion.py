from dotenv import find_dotenv, load_dotenv
from diffusers import StableDiffusionPipeline
import torch

load_dotenv(find_dotenv())

def text2img(prompt, output_path="generated_image.png"):
    # Initialize the pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    
    # If you have a CUDA-capable GPU, uncomment the next line:
    # pipe = pipe.to("cuda")
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(output_path)
    print(f"Image generated and saved to {output_path}")
    
    return output_path

# Example usage
text2img("A serene tropical beach at sunset, with soft golden light reflecting on the calm ocean waves. People are relaxing on the sandy shore, some lying on colorful beach towels, others sitting in lounge chairs under umbrellas. A few are walking along the water's edge, enjoying the gentle breeze. Palm trees sway in the background, and the sky is painted with warm hues of orange and pink. The atmosphere is peaceful and idyllic, perfect for a relaxing vacation.")