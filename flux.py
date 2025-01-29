import os
import gradio as gr
from PIL import Image
from gradio_client import Client


# Set Hugging Face token
os.environ['HF_TOKEN'] = ''

# Create the inference client
client = Client("black-forest-labs/FLUX.1-schnell")

# Set the prompt and parameters for image generation
result = client.predict(
    prompt="A serene tropical beach at sunset, with soft golden light reflecting on the calm ocean waves. People are relaxing on the sandy shore, some lying on colorful beach towels, others sitting in lounge chairs under umbrellas. A few are walking along the water's edge, enjoying the gentle breeze. Palm trees sway in the background, and the sky is painted with warm hues of orange and pink. The atmosphere is peaceful and idyllic, perfect for a relaxing vacation.",
    seed=0,
    randomize_seed=True,
    width=1024,
    height=1024,
    num_inference_steps=4
)

# Open and save the generated image
img = Image.open(result[0])
img.save("generated_image_001.png")

print("Image generated and saved as 'generated_image_001.png'")