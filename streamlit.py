%%writefile app.py
import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import io


# Authorization Header
headers = {"Authorization": "Bearer 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"}

# Set the device to 'cuda' if a GPU is available, else 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache the Stable Diffusion model loading
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return pipe.to(device)

@st.cache_resource
def load_model2():
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return pipe.to(device)

@st.cache_resource
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model.to(device), clip_processor

# Function to query API
def query(API_URL, payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

def infer_model(API_URL):
  image_bytes = query(API_URL,{"inputs": f"{prompt}"})
  example_image = Image.open(io.BytesIO(image_bytes))
  return example_image

# Generate and score an image
def generate_and_score_image(prompt, generator_func):
    try:
        image = generator_func(prompt)
        if not image:
            return None, -1

        # CLIP Scoring
        inputs = clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = clip_model(**inputs)
        similarity_score = outputs.logits_per_image.item()
        return image, similarity_score
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None, -1

# API Query Function
def query2(API_URL, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.text}")
        return None
    return response.content

# Function to generate multi-panel image
def infer_combined_model(prompt):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    try:
        image_bytes = query2(API_URL, {"inputs": prompt})
        if not image_bytes:
            st.error(f"Error: Unable to generate images for the prompt: {prompt}")
            return None
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Failed to generate or decode the image. Error: {e}")
        return None

# Initialize session state for panel descriptions
if "scene_prompts" not in st.session_state:
    st.session_state.scene_prompts = ["Enter description for Panel 1"]
if "num_panels" not in st.session_state:
    st.session_state.num_panels = 1

# Function to add a panel dynamically
def add_panel():
    if st.session_state.num_panels < 4:
        st.session_state.num_panels += 1
        st.session_state.scene_prompts.append(f"Enter description for Panel {st.session_state.num_panels}")


# Load models
pipe1 = load_model()
pipe2 = load_model2()
clip_model, clip_processor = load_clip_model()



# Initialize session state for image count
if "image_count" not in st.session_state:
    st.session_state["image_count"] = 0

# Main App Tabs
tab1, tab2 = st.tabs(["Character Creation", "Comic Scene"])

# Character Creation Tab
with tab1:
    st.title("Character Creation")
    prompt = st.text_area(
        "Enter a prompt to generate character images:",
        "A fierce warrior stands on a rugged cliff, holding a glowing sword, with futuristic armor blending medieval and cyberpunk elements.",
    )

    if st.button("Generate Characters"):
        best_image = None
        highest_score = -1
        best_image_index = -1

        with st.spinner("Generating character images..."):
            for idx, generator in enumerate([pipe1, pipe2, infer_model, infer_model]):
              if idx < 2:  # For the local models
                  image = generator(prompt).images[0]
              else:  # For the Hugging Face API models
                  api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell" if idx == 2 else "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
                  image = generator(api_url)

              # Update image count
              st.session_state['image_count'] += 1

              # Calculate CLIP score for image and prompt
              inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
              outputs = clip_model(**inputs)
              similarity_score = outputs.logits_per_image.item()

              # Update best image if this one has the highest score
              if similarity_score > highest_score:
                  highest_score = similarity_score
                  best_image = image
                  best_image_index = st.session_state['image_count']

              # Display image and similarity score
              st.image(image, caption=f"Image {st.session_state['image_count']} - Score: {similarity_score:.4f}", use_container_width=True)

              # Create download button for the image
              buf = io.BytesIO()
              image.save(buf, format="PNG")
              byte_im = buf.getvalue()
              st.download_button(label="Download Image", data=byte_im, file_name=f"generated_image_{st.session_state['image_count']}.png", mime="image/png")

        # Highlight the best image
        if best_image:
            st.write(f"### Best Image (Image {best_image_index}) - Highest Similarity Score: {highest_score:.4f}")
            st.image(best_image, caption=f"Best Image (Score: {highest_score:.4f})", use_container_width=True)

        st.success("All images generated and evaluated!")


# Comic Scene Tab
with tab2:
  # UI Layout
  st.title("Comic Scene Generator")
  st.write("Describe your comic scene panel by panel. Start with one panel and add up to four.")

  # Display panel description inputs
  for i in range(st.session_state.num_panels):
      st.session_state.scene_prompts[i] = st.text_input(
          f"Panel {i + 1} description",
          st.session_state.scene_prompts[i],
          key=f"panel_{i + 1}"
      )

  # Add Panel Button
  st.button("Add Panel", on_click=add_panel)

  # Generate Comic Button
  if st.button("Generate Comic Scene"):
      with st.spinner("Generating comic panels..."):
          # Combine prompts into a single request
          combined_prompt = (
              f"Illustrate a sequence of {st.session_state.num_panels}-panel scenes: " +
              " ".join([f"Panel {i + 1}: {desc}" for i, desc in enumerate(st.session_state.scene_prompts)])
          )

          # Call the API to generate the combined image
          combined_image = infer_combined_model(combined_prompt)
          if combined_image is None:
              st.error("Failed to generate the comic scene. Please try again later.")
          else:
              # Split the image into individual panels
              panel_width, panel_height = combined_image.size[0], combined_image.size[1] // st.session_state.num_panels
              comic_width = panel_width
              comic_height = panel_height * st.session_state.num_panels
              comic_page = Image.new("RGB", (comic_width, comic_height), "white")

              # Assemble the panels (without borders)
              for i in range(st.session_state.num_panels):
                  panel = combined_image.crop((0, i * panel_height, panel_width, (i + 1) * panel_height))
                  comic_page.paste(panel, (0, i * panel_height))

              # Display the comic page
              st.image(comic_page, caption="Generated Comic Page", use_container_width=True)

              # Create download button for the comic page
              buf = io.BytesIO()
              comic_page.save(buf, format="PNG")
              byte_im = buf.getvalue()
              st.download_button(
                  label="Download Comic Page",
                  data=byte_im,
                  file_name="comic_page.png",
                  mime="image/png"
              )
