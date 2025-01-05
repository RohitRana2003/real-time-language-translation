from flask import Flask, request, jsonify, render_template

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={"/translate": {"origins": "*"}})  # Enable CORS for all routes

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
    "en_XX": "English",
    "hi_IN": "Hindi",
    "ta_IN": "Tamil",
    "bn_IN": "Bengali",
    "gu_IN": "Gujarati",
    "mr_IN": "Marathi",
    "as_IN": "Assamese"
}

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data['text']
    src_lang_code = data['src_lang']
    tgt_lang_code = data['tgt_lang']

    if src_lang_code not in language_codes or tgt_lang_code not in language_codes:
        print("src_lang not in code")
        return jsonify({"error": "Invalid language choice"}), 400

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
        return jsonify({'translated_text': translated_text})
    
    except KeyError as e:
        print(e)
        return jsonify({"error": f"Error: {e}. Ensure the target language code is valid."}), 400

if __name__ == '__main__':
    app.run(debug=True)
