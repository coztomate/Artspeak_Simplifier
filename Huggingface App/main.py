#import libraries
import streamlit as st
from PIL import Image
import io
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_generation import Client
from huggingface_hub import InferenceClient
import config_llm
#for local use
from dotenv import load_dotenv
import os

# Initialize session state variables
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
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
if 'huggingface_key' not in st.session_state:
    st.session_state['huggingface_key'] = ''
if 'message_content_from_caption' not in st.session_state:
    st.session_state['message_content_from_caption'] = ''
if 'message_content_from_simplified_text' not in st.session_state:
    st.session_state['message_content_from_simplified_text'] = ''
if 'mistral_from_caption' not in st.session_state:
    st.session_state['mistral_from_caption'] = ''
if 'mistral_from_simplified' not in st.session_state:
    st.session_state['mistral_from_simplified'] = ''
if 'image_from_caption' not in st.session_state:
    st.session_state['image_from_caption'] = None
if 'image_from_simplified_text' not in st.session_state:
    st.session_state['image_from_simplified_text'] = None
if 'image_from_press_text' not in st.session_state:
    st.session_state['image_from_press_text'] = None
if 'image_from_press_text_from_caption' not in st.session_state:
    st.session_state['image_from_press_text_from_caption'] = None


# Load the tokenizer and simplifier model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-text-simplification")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-small-finetuned-text-simplification")

# Function to simplify text
def simplify_text(input_text):
    # Tokenize and encode the input text
    input_ids = tokenizer.encode("simplify: " + input_text, return_tensors="pt")
    # Generate the simplified text
    output = model.generate(input_ids, min_length=5, max_length=80, do_sample=True)
    # Decode the simplified text
    simplified_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Post-process to ensure the output ends with a complete sentence
    # Find the last period, question mark, or exclamation point
    last_valid_ending = max(simplified_text.rfind('.'), simplified_text.rfind('?'), simplified_text.rfind('!'))
    if last_valid_ending != -1:
        # Ensure the output ends with the last complete sentence
        cleaned_text = simplified_text[:last_valid_ending+1]
    else:
        # No sentence ending found; return the whole text or handle as appropriate
        cleaned_text = simplified_text
    return cleaned_text


# Define the path to example text
example_text_path = "example_text.txt"

# Function to load example text from a file
def load_example_text():
    with open(example_text_path, "r", encoding="utf-8") as file:
        return file.read()
    
# Define the path to your example image
example_image_path = "example.jpg"

# Function to load image from file
def load_image(image_path):
    with open(image_path, "rb") as file:
        # Open the image using PIL
        img = Image.open(file)
        # Load the image data into memory
        img.load()
    return img   



# finds .env file (for running the app locally, otherwise look below)
load_dotenv()

#uncomment the following in case the app is deployed on Hugging Face Spaces and the API key is stored in the secrets
#HF_KEY = st.secrets["hf_key"] 

# Define the API key for the Hugging Face Inference API
HF_KEY = os.getenv("HF_KEY")
st.session_state['huggingface_key'] = HF_KEY
client = InferenceClient(token=HF_KEY)


########################################################################

# Create a Streamlit app
st.title("ARTSPEAK  >  s i m p l i f i e r")

st.markdown("---")

# Create a sub-section for uploading the files
with st.expander("Upload Files"):
        st.markdown("## Upload Text and Image")
        ##### Upload of files
        st.write("Paste your text here or upload example:")
        # Add a button to load example text into the text area
        if st.button('Load Example Text'):
            # Update the session state for user input with the example text
            st.session_state['user_input'] = load_example_text()
        # Add a text input field for user input
        # Directly use session state variable for the value parameter
        user_input = st.text_area("Enter text here", value=st.session_state['user_input'])

        st.markdown("---")
        
        # Load and display example image separately and save for further use
        if st.button("Load Example Image"):
            st.session_state['example_image'] = load_image(example_image_path)
            st.image(st.session_state['example_image'], caption="Example Image")

        # Displaying the file uploader
        uploaded_image = st.file_uploader("Upload an image (jpg or png)", type=["jpg", "png"])


st.markdown("---")

#### Simplifier and Image Caption
with st.expander("Simplify Text and Image"):
    st.markdown("## 'Simplify' Text and Image")

    ## Text simplifier
    if st.button("Simplify the Input Text"):
        if user_input:
            simplified_text = simplify_text(user_input)
            st.session_state['simplified_text'] = simplified_text
        else:
            st.warning("Please enter text in the input field before clicking 'Save'")

    # Display the simplified text from session state
    if st.session_state['simplified_text']:
        st.write(st.session_state['simplified_text'])

    ## Get new caption
    # Button to get new caption
    if st.button("Get New Caption for Image"):
        # Initialize image data variable
        image_data = None

        # Check if the user has uploaded an image
        if uploaded_image is not None:
            image_data = uploaded_image.getvalue()
        # If not, check if the example image has been loaded
        elif 'example_image' in st.session_state:
            # Convert PIL Image to bytes for example image
            buffer = io.BytesIO()
            st.session_state['example_image'].save(buffer, format="PNG")
            buffer.seek(0)
            image_data = buffer.getvalue()

        # If we have image data, get the caption
        if image_data is not None:
            try:
                # Generate the caption (make sure to send the image in the correct format expected by your API)
                caption = client.image_to_text(image_data)
                # Update the session state
                st.session_state['new_caption'] = caption
                st.write(st.session_state['new_caption'])

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image or load the example image before clicking 'Get New Caption for Image'")


st.markdown("---")

########################################################################

with st.expander("Press Text Generation"):
    st.markdown("## Generate New Presstext for an Exhibition")

    # Define radio button options
    option = st.radio(
        "Choose a Language Model:",
        ('Mistral 8x7B', 'GPT-3.5 Turbo'))

    # Conditional logic based on radio button choice
    if option == 'Mistral 8x7B':
        st.header("Mistral 8x7B")

        ############
        ###Mistral##
        ############

        headers = {"Authorization": f"Bearer {st.session_state['huggingface_key']}"}

        client_mistral = Client(
            config_llm.API_URL,
            headers=headers,
        )

        def run_single_input(
            message: str,
            system_prompt: str = config_llm.DEFAULT_SYSTEM_PROMPT,
            max_new_tokens: int = config_llm.MAX_NEW_TOKENS,
            temperature: float = config_llm.TEMPERATURE,
            top_p: float = config_llm.TOP_P
        ) -> str:
            """
            Run the model for a single input and return a single output.
            """
            prompt = f"{system_prompt}\n\nUser: {message.strip()}\n"

            generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            )
            stream = client_mistral.generate_stream(prompt, **generate_kwargs)
            output = ""
            for response in stream:
                if any([end_token in response.token.text for end_token in [config_llm.EOS_STRING, config_llm.EOT_STRING]]):
                    break  # Stop at the first end token
                else:
                    output += response.token.text

            return output.strip()  # Return the complete output
        

        # Button to generate press text from new caption from Mistral
        if st.button("Generate Press Text from New Image Caption with Mistral"):
            if st.session_state['new_caption']:
                try:
                    st.session_state['mistral_from_caption'] = run_single_input(st.session_state['new_caption'], config_llm.DEFAULT_SYSTEM_PROMPT)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please ensure a caption is generated.")

        # Display the generated press text from new caption
        if st.session_state['mistral_from_caption']:
            st.write("Generated Press Text from New Caption of Artwork:")
            st.write(st.session_state['mistral_from_caption'])

        # Button to generate press text from simplified text
        if st.button("Generate Press Text from Simplified Text with Mistral"):
            if st.session_state['simplified_text']:
                try:
                    st.session_state['mistral_from_simplified'] = run_single_input(st.session_state['simplified_text'], config_llm.DEFAULT_SYSTEM_PROMPT)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please ensure simplified text is available.")

        # Display the generated press text from simplified text
        if st.session_state['mistral_from_simplified']:
            st.write("Generated Press Text from Simplified Text:")
            st.write(st.session_state['mistral_from_simplified'])
        
    elif option == 'GPT-3.5 Turbo':
        st.header("GPT-3.5")

        ##########
        ##OpenAI##
        ##########

        # Define the OpenAI API key from .env file for running locally, otherwise use the following lines
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        st.session_state['openai_api_key'] = OPENAI_API_KEY

        # uncomment the following in case the app is deployed on Hugging Face Spaces or Streamlit
        # Add a text input for the OpenAI API key
        #api_key_input = st.text_input("Enter your OpenAI API key to continue", type="password")

        # Button to save the API key
        #if st.button('Save API Key'):
        #    st.session_state['openai_api_key'] = api_key_input
        #    st.success("API Key saved temporarily for this session.")
        #st.write("-  -  -")

        # Function to get completion from OpenAI API
        def get_openai_completion(api_key, prompt_message):
            client = OpenAI(api_key=api_key,)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=config_llm.MAX_NEW_TOKENS,
                temperature = config_llm.TEMPERATURE,
                top_p = config_llm.TOP_P,
                messages=[
                    {"role": "system", "content": config_llm.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_message}
                ]
            )
            return completion.choices[0].message.content

        # Button to generate press text from new caption
        if st.button("Generate Press Text from New Image Caption with GPT"):
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
        if st.button("Generate Press Text from Simplified Text with GPT"):
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

########################################################################

## Image Generation Interface

with st.expander("Image Generation"):
    st.markdown("## Generate new Images from Texts")
    # Button to generate image from new caption
    if st.button("Generate Image from New Caption of Artwork"):
        if st.session_state['new_caption']:
            prompt_caption = f"contemporary art of {st.session_state['new_caption']}"
            st.session_state['image_from_caption'] = client.text_to_image(prompt_caption, model="prompthero/openjourney-v4")

    # Display the image generated from new caption
    if st.session_state['image_from_caption'] is not None:
        st.image(st.session_state['image_from_caption'], caption="Image from New Caption", use_column_width=True)

    # Button to generate image from simplified text
    if st.button("Generate Image from Simplified Text"):
        if st.session_state['simplified_text']:
            prompt_summary = f"contemporary art of {st.session_state['simplified_text']}"
            st.session_state['image_from_simplified_text'] = client.text_to_image(prompt_summary, model="prompthero/openjourney-v4")

    # Display the image generated from simplified text
    if st.session_state['image_from_simplified_text'] is not None:
        st.image(st.session_state['image_from_simplified_text'], caption="Image from Simplified Text", use_column_width=True)

    # Button to generate image from press text from simplified text

    if st.button("Generate Image from new Press Text from Simplified Text"):
        text_to_use_simp = None

        # Check which variable is available and set it to text_to_use
        if 'mistral_from_simplified' in st.session_state and st.session_state['mistral_from_simplified']:
            text_to_use_simp = st.session_state['mistral_from_simplified']
        elif 'message_content_from_simplified_text' in st.session_state and st.session_state['message_content_from_simplified_text']:
            text_to_use_simp = st.session_state['message_content_from_simplified_text']

        # Use the available text to generate the image
        if text_to_use_simp:
            # Check for length of the text and truncate if necessary
            if len(text_to_use_simp) > 509:  # Adjust based on your model's max length (512-3)
                text_to_use_simp = text_to_use_simp[:509]  # Truncate the text

            prompt_press_text_simple = f"contemporary art of {text_to_use_simp}"
            try:
                st.session_state['image_from_press_text'] = client.text_to_image(prompt_press_text_simple, model="prompthero/openjourney-v4")
            except Exception as e:
                st.error("Failed to generate image: " + str(e))
        else:
            st.error("First generate a press text from summary.")

    # Display the image generated from press text from simplified text
    if 'image_from_press_text' in st.session_state and st.session_state['image_from_press_text'] is not None:
        st.image(st.session_state['image_from_press_text'], 
                caption="Image from Press Text from simplified Text", 
                use_column_width=True)

    # Button to generate image from press text from caption
    if st.button("Generate Image from new Press Text from new Caption"):
        # Initialize the variable
        text_to_use_cap = None
        # Check which variable is available and set it to text_to_use
        if 'mistral_from_caption' in st.session_state and st.session_state['mistral_from_caption']:
            text_to_use_cap = st.session_state['mistral_from_caption']
        elif 'message_content_from_caption' in st.session_state and st.session_state['message_content_from_caption']:
            text_to_use_cap = st.session_state['message_content_from_caption']

        # Use the available text to generate the image
        if text_to_use_cap:
            # Check for length of the text and truncate if necessary
            if len(text_to_use_cap) > 509:  # Adjust based on your model's max length
                text_to_use_cap = text_to_use_cap[:509]  # Truncate the text

            prompt_press_text_caption = f"contemporary art of {text_to_use_cap}"
            try:
                st.session_state['image_from_press_text_from_caption'] = client.text_to_image(prompt_press_text_caption, model="prompthero/openjourney-v4")
            except Exception as e:
                st.error("Failed to generate image: " + str(e))
        else:
            st.error("First generate a press text from summary.")

    # Display the image generated from press text from caption
    if st.session_state['image_from_press_text_from_caption'] is not None:
        st.image(st.session_state['image_from_press_text_from_caption'], 
                caption="Image from Press Text from new Caption", 
                use_column_width=True)
        
st.markdown("---")