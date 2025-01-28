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
text2img("a group of people posing for a picture in front of a picture frame")