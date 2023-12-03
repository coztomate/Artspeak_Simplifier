#import libraries
import streamlit as st
from PIL import Image
import io
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torchvision import transforms
import open_clip

# Initialize session state variables
if 'simplified_text' not in st.session_state:
    st.session_state['simplified_text'] = ''
if 'new_caption' not in st.session_state:
    st.session_state['new_caption'] = ''

#### Load the Models
# Specify the model and tokenizer names or paths
model_name = "mrm8488/t5-small-finetuned-text-simplification"
tokenizer_name = "mrm8488/t5-small-finetuned-text-simplification"

# Load models only once in session state
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    st.session_state['model'] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(tokenizer_name)
    st.session_state['simplifier'] = pipeline("text2text-generation", model=st.session_state['model'], tokenizer=st.session_state['tokenizer'])

# Use the model from session state
simplifier = st.session_state['simplifier']

#clip model
model_clip, _, transform_clip = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )

# Create a Streamlit app
st.title("ARTSPEAK")

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
    st.write("Simplified Text:")
    st.write(st.session_state['simplified_text'])

####Get new caption

# Function to generate a caption for the uploaded image
def generate_caption(image_path):
    
    # Load and preprocess the uploaded image
    im = Image.open(image_path).convert("RGB")
    im = transform_clip(im).unsqueeze(0)

    # Generate a caption for the image
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model_clip.generate(im)

    new_caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2]
    return new_caption

####Get new caption
if st.button("Get Caption"):
    if uploaded_image is not None:
        # Generate the caption
        caption = generate_caption(uploaded_image)
        # Update the session state
        st.session_state['new_caption'] = caption
    else:
        st.warning("Please upload an image before clicking 'Get Caption'")

# Display the new caption from session state
if st.session_state['new_caption']:
    st.write("New Caption for Artwork:")
    st.write(st.session_state['new_caption'])
