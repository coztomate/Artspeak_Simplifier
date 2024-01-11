# ARTSPEAK Simplifier
ARTSPEAK Simplifier is a Streamlit app designed to simplify texts and generate contemporary art images based on simplified texts or new captions. Adopting a playful and humorous approach on the topic of International Art English (IAE), a distinct form of English prevalent in the contemporary art world characterized by its specialized vocabulary and complex structures, it ignites a discussion on the accessibility of such language in the art discourse. Users can upload press text and an accompanying image from a contemporary art exhibition. The app “simplifies” both the text and the image into a concise sentence each. These simplifications serve as a foundation to generate new, intricate press texts in authentic IAE style. Furthermore, the app creates four unique contemporary art images based on these simplifications and the newly formed texts.

By leveraging a variety of Large Language Models (LLMs) and AI services, ARTSPEAK Simplifier performs text simplification, text generation, and image generation, demonstrating the remarkable capabilities of HuggingFace and OpenAI APIs. This project showcases how inference endpoints (or inference APIs) can be combined and utilized to build complex applications and features.

## Features
- Text Simplification: Simplify complex texts using T5 model fine-tuned for text simplification.
- Example Loading: Load example texts and images for quick demonstration.
- Image Captioning: Generate new captions for uploaded images.
- Press Text Generation: Create new press text for exhibitions using either Mixtral or GPT-3.5.
- Image Generation: Generate contemporary art images from texts, simplified texts, or new press texts.

## Usage
To get started with the ARTSPEAK Simplifier:
1. Run Notebooks: Clone the repository and explore the notebooks provided in the repo. Please note that some features like Mixtral are not be included yet.
2. Hugging Face App:
    - Navigate to the Huggingface App folder.
    - Install the required libraries using `pip install -r requirements.txt`.
    - Run the app using `streamlit run main.py`.
    - The app should now be running on your local server. Follow the instructions on the interface to upload texts and images, simplify texts, and generate new press texts and images.

## How It Works
### Text Simplification
T5 Model: Tokenize and encode the input text. Generate the simplified text using a pre-trained T5 model. Post-process the output to ensure it ends with a complete sentence.
### Image Captioning
BLIP: Upload an image or use the provided example. The image is sent to the BLIP model, which generates a caption.
### Press Text Generation
Mixtral 8x7B and GPT-3.5 Turbo: Choose between these models to generate press texts. The selected model generates press text based on the new image caption or simplified text.
### Image Generation
Stable Diffusion (stable-diffusion-v1-4): Generate images from new captions, simplified texts, or new press texts. The app sends a prompt to the Stable Diffusion model which generates a contemporary art image.

## License
Distributed under the MIT License. See LICENSE for more information.