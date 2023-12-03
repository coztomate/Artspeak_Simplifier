#import libraries
import streamlit as st
from PIL import Image
import io
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch
from torchvision import transforms
import open_clip
from openai import OpenAI
import openai
from diffusers import StableDiffusionPipeline

# Initialize session state variables
if 'simplified_text' not in st.session_state:
    st.session_state['simplified_text'] = ''
if 'new_caption' not in st.session_state:
    st.session_state['new_caption'] = ''
if 'model_clip' not in st.session_state:
    st.session_state['model_clip'] = None
if 'transform_clip' not in st.session_state:
    st.session_state['transform_clip'] = None
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''
if 'message_content_from_caption' not in st.session_state:
    st.session_state['message_content_from_caption'] = ''
if 'message_content_from_simplified_text' not in st.session_state:
    st.session_state['message_content_from_simplified_text'] = ''
if 'image_from_caption' not in st.session_state:
    st.session_state['image_from_caption'] = None
if 'image_from_simplified_text' not in st.session_state:
    st.session_state['image_from_simplified_text'] = None
if 'image_from_press_text' not in st.session_state:
    st.session_state['image_from_press_text'] = None

######loading models########

####loading simplifier model#####
# Define model and tokenizer names for the text simplification model
model_name = "mrm8488/t5-small-finetuned-text-simplification"
tokenizer_name = "mrm8488/t5-small-finetuned-text-simplification"

# Load models only once in session state
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    st.session_state['model'] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(tokenizer_name)
    st.session_state['simplifier'] = pipeline("text2text-generation", model=st.session_state['model'], tokenizer=st.session_state['tokenizer'])

# Use the model from session state
simplifier = st.session_state['simplifier']

####loading clip model#####
# Function to load the CLIP model
def load_clip_model():
    model_clip, _, transform_clip = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    return model_clip, transform_clip

if 'loaded_clip_model' not in st.session_state or 'loaded_transform_clip' not in st.session_state:
    st.session_state['loaded_clip_model'], st.session_state['loaded_transform_clip'] = load_clip_model()

# Function to generate a caption using the preloaded CLIP model
def generate_caption(image_path):
    im = Image.open(image_path).convert("RGB")
    im = st.session_state['loaded_transform_clip'](im).unsqueeze(0)

    # Generate a caption for the image
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = st.session_state['loaded_clip_model'].generate(im)

    new_caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2]
    return new_caption

###loading diffusion model
# Function to load the Stable Diffusion model
def load_diffusion_model():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

# Initialize the model at the start and store it in the session state
if 'loaded_model' not in st.session_state:
    st.session_state['loaded_model'] = load_diffusion_model()

# Function to generate an image using the preloaded model
def generate_image(prompt):
    image = st.session_state['loaded_model'](prompt).images[0]
    return image

################################################

# Create a Streamlit app
st.title("ARTSPEAK  >  s i m p l i f i e r")

st.markdown("---")

##### Upload of files
# Add a text input field for user input
user_input = st.text_area("Enter text here")
 
# Add an upload field to the app for image files (jpg or png)
uploaded_image = st.file_uploader("Upload an image (jpg or png)", type=["jpg", "png"])

#### Display of files
# Create a sub-section
with st.expander("Display of Uploaded Files"):
    st.write("These are you uploaded files:")
    # Check if a file was uploaded
    if user_input is not None:
        # Display file information
        st.write("Original Text:")
        st.write(user_input)

    # Check if an image was uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

st.markdown("---")

####Get summary
if st.button("Simplify"):
    if user_input:
        simplified_text = simplifier(user_input, min_length=20, max_length=50, do_sample=True)
        # Update the session state
        st.session_state['simplified_text'] = simplified_text[0]['generated_text']
    else:
        st.warning("Please enter text in the input field before clicking 'Save'")

# Display the simplified text from session state
if st.session_state['simplified_text']:
    st.write("Simplified Original Text:")
    st.write(st.session_state['simplified_text'])

st.markdown("---")

####Get new caption
# Modify the 'Get Caption' button section
if st.button("Get New Caption"):
    if uploaded_image is not None:
        # Generate the caption
        caption = generate_caption(uploaded_image)
        # Update the session state
        st.session_state['new_caption'] = caption
    else:
        st.warning("Please upload an image before clicking 'Get Caption'")

# Display the new caption from session state
if st.session_state['new_caption']:
    st.write("New Caption for this Artwork:")
    st.write(st.session_state['new_caption'])

st.markdown("---")

#######
#OpenAI API
#######
# Add a text input for the OpenAI API key
api_key_input = st.text_input("Enter your OpenAI API key if you want more", type="password")

# Button to save the API key
if st.button('Save API Key'):
    st.session_state['openai_api_key'] = api_key_input
    st.success("API Key saved temporarily for this session.")

st.markdown("---")

# Function to get completion from OpenAI API
def get_openai_completion(api_key, prompt_message):
    client = OpenAI(api_key=api_key,)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "I give a short description of an artwork. Please write a complex press text for an exhibiton in international art english dealing with post-colonialism, identity politics, military industrial complex and queerness through the language of writerts like Ranciere, Deleuze, Trevor Paglen, Hito Steyerl, Slavoy Zizek, Claire Fontane, Michel Foucault, Donna Harraway and Paul Preciado. Without doing too many name drops. Just output the press text and not surrounding or explaining messages with it."},
            {"role": "user", "content": prompt_message}
        ]
    )
    return completion.choices[0].message.content

# Button to generate press text from new caption
if st.button("Generate Press Text from New Caption"):
    if st.session_state['new_caption'] and st.session_state['openai_api_key']:
        try:
            st.session_state['message_content_from_caption'] = get_openai_completion(st.session_state['openai_api_key'], st.session_state['new_caption'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please ensure a caption is generated and an API key is entered.")

# Display the generated press text from new caption
if st.session_state['message_content_from_caption']:
    st.write("Generated Press Text from New Caption of Artwork:")
    st.write(st.session_state['message_content_from_caption'])

# Button to generate press text from simplified text
if st.button("Generate Press Text from Simplified Text"):
    if st.session_state['simplified_text'] and st.session_state['openai_api_key']:
        try:
            st.session_state['message_content_from_simplified_text'] = get_openai_completion(st.session_state['openai_api_key'], st.session_state['simplified_text'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please ensure simplified text is available and an API key is entered.")

# Display the generated press text from simplified text
if st.session_state['message_content_from_simplified_text']:
    st.write("Generated Press Text from Simplified Text:")
    st.write(st.session_state['message_content_from_simplified_text'])

st.markdown("---")

############
##Diffusor##
############


# Example button to generate image from new caption
if st.button("Generate Image from New Caption of Artwork"):
    if st.session_state['new_caption']:
        prompt_caption = f"contemporary art of {st.session_state['new_caption']}"
        st.session_state['image_from_caption'] = generate_image(prompt_caption)

# Display the image generated from new caption
if st.session_state['image_from_caption'] is not None:
    st.image(st.session_state['image_from_caption'], caption="Image from New Caption", use_column_width=True)

# Button to generate image from simplified text
if st.button("Generate Image from Simplified Text"):
    if st.session_state['simplified_text']:
        prompt_summary = f"contemporary art of {st.session_state['simplified_text']}"
        st.session_state['image_from_simplified_text'] = generate_image(prompt_summary)

# Display the image generated from simplified text
if st.session_state['image_from_simplified_text'] is not None:
    st.image(st.session_state['image_from_simplified_text'], caption="Image from Simplified Text", use_column_width=True)

# Button to generate image from press text
if st.button("Generate Image from new Press Text"):
    if st.session_state['message_content_from_simplified_text']:
        prompt_press_text = f"contemporary art of {st.session_state['message_content_from_simplified_text']}"
        st.session_state['image_from_press_text'] = generate_image(prompt_press_text)

# Display the image generated from press text
if st.session_state['image_from_press_text'] is not None:
    st.image(st.session_state['image_from_press_text'], caption="Image from Press Text", use_column_width=True)


st.markdown("---")