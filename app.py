import streamlit as st
import torch
import io

from diffusers import StableDiffusionPipeline

# --------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="PromptCraft AI",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 PromptCraft AI")
st.subheader("Transform your imagination into AI-generated images")

st.write(
    "Enter a description, select an art style "
    "and let Stable Diffusion create your image."
)

# --------------------------------
# 2. LOAD STABLE DIFFUSION MODEL
# --------------------------------

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        use_safetensors=True
    )

    model = model.to(device)

    return model, device


# --------------------------------
# 3. PROMPT ENGINEERING MODULE
# --------------------------------

def enhance_prompt(user_prompt, style):

    styles = {
        "Realistic": "photorealistic, natural lighting, DSLR photography",
        "Anime": "anime illustration, vibrant colors, detailed characters",
        "3D Art": "3D rendering, realistic textures, cinematic lighting",
        "Digital Art": "digital painting, creative composition, vibrant colors",
        "Oil Painting": "oil painting, artistic brush strokes, rich colors",
        "Cinematic": "cinematic photography, dramatic lighting, movie scene"
    }

    enhanced = (
        f"{user_prompt}, "
        f"{styles[style]}, "
        "highly detailed, sharp focus, "
        "professional composition"
    )

    negative = (
        "blurry, distorted, low quality, "
        "bad anatomy, watermark, text"
    )

    return enhanced, negative


# --------------------------------
# 4. SIDEBAR SETTINGS
# --------------------------------

st.sidebar.title("⚙️ Generation Settings")

style = st.sidebar.selectbox(
    "Choose Image Style",
    [
        "Realistic",
        "Anime",
        "3D Art",
        "Digital Art",
        "Oil Painting",
        "Cinematic"
    ]
)

steps = st.sidebar.slider(
    "Inference Steps",
    min_value=15,
    max_value=40,
    value=25
)

guidance = st.sidebar.slider(
    "Guidance Scale",
    min_value=1.0,
    max_value=15.0,
    value=7.5,
    step=0.5
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=999999,
    value=42
)

# --------------------------------
# 5. CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":
            st.write(message["content"])

        else:
            st.write("**Enhanced Prompt:**")
            st.write(message["enhanced"])

            st.image(
                message["image"],
                caption="Generated Image",
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Image",
                data=message["image_bytes"],
                file_name="generated_image.png",
                mime="image/png",
                key=message["download_key"]
            )


# --------------------------------
# 6. USER PROMPT INPUT
# --------------------------------

user_prompt = st.chat_input(
    "Describe the image you want to generate..."
)


# --------------------------------
# 7. GENERATE IMAGE
# --------------------------------

if user_prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Loading AI model..."):

                pipe, device = load_model()

            enhanced, negative = enhance_prompt(
                user_prompt,
                style
            )

            st.write("**Enhanced Prompt:**")
            st.write(enhanced)

            with st.spinner("Creating your image..."):

                generator = torch.Generator(
                    device=device
                ).manual_seed(int(seed))

                result = pipe(
                    prompt=enhanced,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator
                )

                image = result.images[0]

            st.success("Image generated successfully!")

            st.image(
                image,
                caption="Your AI Generated Image",
                use_container_width=True
            )

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            download_key = f"download_{len(st.session_state.messages)}"

            st.download_button(
                "⬇️ Download Image",
                data=image_bytes,
                file_name="generated_image.png",
                mime="image/png",
                key=download_key
            )

            st.session_state.messages.append({
                "role": "assistant",
                "enhanced": enhanced,
                "image": image,
                "image_bytes": image_bytes,
                "download_key": download_key
            })

        except Exception as e:

            st.error(f"Image generation failed: {e}")