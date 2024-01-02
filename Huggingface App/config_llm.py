# Configuration and Constants

# Defaults for Mistral/OpenAI
DEFAULT_SYSTEM_PROMPT = "You will be given a very short description of a contemporary artwork. Please create a complex exhibition press text based on the given artwork description using international art english dealing with post-colonialism, military industrial complex, anthropocene, identity politics and queerness through the language of Rancière, Fontane, Paglen, Deleuze, Steyerl, Spivak, Preciado, Žižek, Foucault and Harraway. Avoid excessive namedropping. Just output press text without explaining your actions."
EOS_STRING = "</s>"
EOT_STRING = "<EOT>"

# Mistral Model Configuration
model_id_mistral = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{model_id_mistral}"

#model parameters
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.8