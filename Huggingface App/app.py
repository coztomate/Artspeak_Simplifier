#import libraries
import streamlit as st
from PIL import Image
import io
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from huggingface_hub import InferenceClient
client = InferenceClient()

# load the simplifier model
# Load the tokenizer and model (do this outside the function for efficiency)
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-text-simplification")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-small-finetuned-text-simplification")

def simplify_text(input_text):
    # Tokenize and encode the input text
    input_ids = tokenizer.encode("simplify: " + input_text, return_tensors="pt")

    # Generate the simplified text
    output = model.generate(input_ids, min_length=20, max_length=50, do_sample=True)
    
    # Decode the simplified text
    simplified_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove or replace unwanted tokens like "SEP>"
    cleaned_text = simplified_text.replace("SEP>", "")

    return cleaned_text


# Initialize session state variables
if 'summary_text' not in st.session_state:
    st.session_state['summary_text'] = ''
if 'simplified_text' not in st.session_state:
    st.session_state['simplified_text'] = ''
if 'new_caption' not in st.session_state:
    st.session_state['new_caption'] = None
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
if 'image_from_press_text_from_caption' not in st.session_state:
    st.session_state['image_from_press_text_from_caption'] = None


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
if st.button("Summarize"):
    if user_input:
        summarized_text = client.summarization(user_input)
        st.session_state['summary_text'] = summarized_text
    else:
        st.warning("Please enter text in the input field before clicking 'Save'")

# Display the simplified text from session state
if st.session_state['summary_text']:
    st.write("Summary Original Text:")
    st.write(st.session_state['summary_text'])

st.markdown("---")

if st.button("Simplify"):
    if user_input:
        simplified_text = simplify_text(user_input)
        st.session_state['simplified_text'] = simplified_text
    else:
        st.warning("Please enter text in the input field before clicking 'Save'")

# Display the simplified text from session state
if st.session_state['simplified_text']:
    st.write("Simplify Original Text:")
    st.write(st.session_state['simplified_text'])

st.markdown("---")

####Get new caption
# Modify the 'Get Caption' button section
if st.button("Get New Caption for Uploaded Image"):
    if uploaded_image is not None:
        try:
            # Convert the uploaded file to bytes
            image_bytes = uploaded_image.getvalue()

            # Generate the caption (make sure to send the image in the correct format expected by your API)
            caption = client.image_to_text(image_bytes)
            
            # Update the session state
            st.session_state['new_caption'] = caption

        except Exception as e:
            st.error(f"An error occurred: {e}")
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
            {"role": "system", "content": "I give a short description of an artwork. Please create a complex exhibition press text based on the given artwork description using international art english dealing with post-colonialism, military industrial complex, anthropocene, identity politics and queerness through the language of Rancière, Fontane, Paglen, Deleuze, Steyerl, Spivak, Preciado, Žižek, Foucault and Harraway. Avoid excessive namedropping. Just output press text without explaining your actions."},
            {"role": "user", "content": prompt_message}
        ]
    )
    return completion.choices[0].message.content

# Button to generate press text from new caption
if st.button("Generate Press Text from New Image Caption"):
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
        st.session_state['image_from_caption'] = client.text_to_image(prompt_caption)

# Display the image generated from new caption
if st.session_state['image_from_caption'] is not None:
    st.image(st.session_state['image_from_caption'], caption="Image from New Caption", use_column_width=True)

# Button to generate image from simplified text
if st.button("Generate Image from Simplified Text"):
    if st.session_state['simplified_text']:
        prompt_summary = f"contemporary art of {st.session_state['simplified_text']}"
        st.session_state['image_from_simplified_text'] = client.text_to_image(prompt_summary)

# Display the image generated from simplified text
if st.session_state['image_from_simplified_text'] is not None:
    st.image(st.session_state['image_from_simplified_text'], caption="Image from Simplified Text", use_column_width=True)

# Button to generate image from press text from simplified text
if st.button("Generate Image from new Press Text from Simplified Text"):
    if st.session_state['message_content_from_simplified_text']:
        prompt_press_text = f"contemporary art of {st.session_state['message_content_from_simplified_text']}"
        st.session_state['image_from_press_text'] = client.text_to_image(prompt_press_text)

# Display the image generated from press text from simplified text
if st.session_state['image_from_press_text'] is not None:
    st.image(st.session_state['image_from_press_text'], 
             caption="Image from Press Text from simplified Text", 
             use_column_width=True)

# Button to generate image from press text from caption
if st.button("Generate Image from new Press Text from new Caption"):
    if st.session_state['message_content_from_caption']:
        prompt_press_text_caption = f"contemporary art of {st.session_state['message_content_from_caption']}"
        st.session_state['image_from_press_text_from_caption'] = client.text_to_image(prompt_press_text_caption)

# Display the image generated from press text from caption
if st.session_state['image_from_press_text_from_caption'] is not None:
    st.image(st.session_state['image_from_press_text_from_caption'], 
             caption="Image from Press Text from new Caption", 
             use_column_width=True)

st.markdown("---")