# PromptCraft AI

An intelligent text-to-image generation application using Stable Diffusion 1.5 and prompt engineering.

## Features
- Text-to-image generation
- Prompt enhancement
- Six artistic styles: Realistic, Anime, 3D Art, Digital Art, Oil Painting and Cinematic
- Adjustable inference steps and guidance scale
- Seed control for reproducibility
- PNG image download
- CLIP-based image-text similarity evaluation

## Technologies
- Python
- Streamlit
- PyTorch
- Stable Diffusion 1.5
- Hugging Face Diffusers
- CLIP
- Google Colab

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Evaluation
The project was evaluated using eight text prompts and CLIP similarity scores. The experiment also compared image generation time for original and enhanced prompts.

## Model
Stable Diffusion 1.5:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

## Note
Image generation requires significant computing resources. GPU acceleration is recommended.
