from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Load the mBART model and tokenizer
print("Loading the mBART model and tokenizer...")
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded successfully.\n")

# Language mapping
language_codes = {
    "hindi": "hi_IN",
    "tamil": "ta_IN",
    "bengali": "bn_IN",
    "english": "en_XX",
    "gujarati": "gu_IN",  # Added Gujarati language code
    "marathi": "mr_IN",   # Added Marathi language code
    "assamese": "as_IN"   # Added Assamese language code
}

print("Languages available for translation using mBART:")
for lang, code in language_codes.items():
    print(f"{lang.capitalize()}: {code}")

while True:
    # User inputs
    print("\nEnter source and target languages:")
    src_lang = input("Source language (Hindi/Tamil/Bengali/English/Gujarati/Marathi/Assamese): ").strip().lower()
    tgt_lang = input("Target language (Hindi/Tamil/Bengali/English/Gujarati/Marathi/Assamese): ").strip().lower()

    if src_lang not in language_codes or tgt_lang not in language_codes:
        print("Invalid language choice. Please select from the available options.")
        continue

    # Setting source and target language codes
    src_lang_code = language_codes[src_lang]
    tgt_lang_code = language_codes[tgt_lang]

    # Text to translate
    text = input("Enter text to translate (type 'exit' to quit): ").strip()
    if text.lower() == 'exit':
        print("Exiting the translator. Goodbye!")
        break

    # Debugging: Print the language codes
    print(f"Source Language Code: {src_lang_code}, Target Language Code: {tgt_lang_code}")

    # Translation
    tokenizer.src_lang = src_lang_code
    encoded_input = tokenizer(text, return_tensors="pt").to(device)

    try:
        # Force the model to generate in the target language
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code]
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        print(f"\nTranslated Text ({src_lang.capitalize()} to {tgt_lang.capitalize()}): {translated_text}")
    except KeyError as e:
        print(f"Error: {e}. Ensure the target language code is valid.")
