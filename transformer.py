"""
MBART Multilingual Translation Script
-------------------------------------
This script utilizes Facebook's mBART (Multilingual BART) model for multilingual text translation.
It supports translation between Hindi, Tamil, Bengali, and English.
"""

# Importing necessary libraries
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import torch

# Initialize the mBART model and tokenizer
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
print("Loading mBART model and tokenizer...")
tokenizer = MBart50Tokenizer.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model and tokenizer loaded successfully.\n")

# Language codes supported by this implementation
LANGUAGE_CODES = {
    "Hindi": "hi_IN",
    "Tamil": "ta_IN",
    "Bengali": "bn_IN",
    "English": "en_XX",
}

def display_languages():
    """
    Display the list of supported languages and their corresponding codes.
    """
    print("Supported Languages:")
    for language, code in LANGUAGE_CODES.items():
        print(f"  {language}: {code}")

def translate_text(text, src_lang, tgt_lang):
    """
    Translate text using the mBART model.
    
    Parameters:
    - text: The input text to be translated.
    - src_lang: Source language code (e.g., 'hi_IN').
    - tgt_lang: Target language code (e.g., 'en_XX').
    
    Returns:
    - The translated text as a string.
    """
    try:
        # Validate input languages
        if src_lang not in LANGUAGE_CODES.values() or tgt_lang not in LANGUAGE_CODES.values():
            raise ValueError(f"Invalid language pair: {src_lang} -> {tgt_lang}")
        
        # Set source language for the tokenizer
        tokenizer.src_lang = src_lang
        
        # Tokenize input text
        encoded_text = tokenizer(text, return_tensors="pt")
        
        # Generate translation
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
        
        # Decode the generated tokens
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """
    Main function for interactive translation.
    """
    display_languages()
    print("\nWelcome to the Multilingual Translation Program!\n")
    
    while True:
        # Prompt user for source and target languages
        src_lang_name = input("Enter Source Language (e.g., Hindi, Tamil, English): ").capitalize()
        tgt_lang_name = input("Enter Target Language (e.g., Bengali, English, Hindi): ").capitalize()
        
        # Get language codes
        src_lang_code = LANGUAGE_CODES.get(src_lang_name)
        tgt_lang_code = LANGUAGE_CODES.get(tgt_lang_name)
        
        # Validate language inputs
        if not src_lang_code or not tgt_lang_code:
            print("Invalid language selection. Please try again.")
            continue
        
        # Input the text to be translated
        text = input(f"Enter text to translate from {src_lang_name} to {tgt_lang_name} (or type 'exit' to quit): ")
        if text.lower() == "exit":
            print("Exiting the translation program. Goodbye!")
            break
        
        # Translate the text
        translation = translate_text(text, src_lang_code, tgt_lang_code)
        
        # Display the translation
        if translation:
            print(f"\nTranslated Text ({src_lang_name} -> {tgt_lang_name}): {translation}\n")
        else:
            print("Translation failed. Please try again.\n")

if __name__ == "__main__":
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        model.to(device)
        print("Model moved to GPU for faster translation.\n")
    
    # Start the translation program
    main()
