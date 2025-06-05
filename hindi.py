"""# Import necessary libraries
from transformers import MarianMTModel, MarianTokenizer
import stanza
from googletrans import Translator
import string

# Define the model name for Hindi to English translation (Helsinki-NLP)
model_name = 'Helsinki-NLP/opus-mt-hi-en'

# Load the tokenizer and model for MarianMT
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize the Hindi NLP model using stanza
stanza.download('hi')  # Download Hindi model
hi_nlp = stanza.Pipeline('hi')

stop_words_hi = [
    'और', 'में', 'के', 'है', 'यह', 'को', 'कि', 'का', 'से', 'लिए', 'हैं', 'पर', 'हम', 'आप', 'वह', 'इन', 'ऐसा', 'कुछ', 'हो', 'होगा', 'जो'
]

# Translator instance for translation
translator = Translator()

# Define the MarianMT translation function
def translate(text):
    # Tokenize the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    
    # Perform the translation
    translation = model.generate(**tokenized_text)
    
    # Decode the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    
    return translated_text

# Converts Hindi text to sentence list using Stanza
def convert_to_sentence_list_hindi(text):
    doc = hi_nlp(text)
    sent_list = []
    for sentence in doc.sentences:
        sent_list.append(sentence.text)
    return sent_list

# Converts sentence to words array for each sentence
def convert_to_word_list_hindi(sentences):
    word_list = []
    for sentence in sentences:
        # Iterate through each sentence's tokens and extract the word text
        word_list.append([token.text for token in hi_nlp(sentence).sentences[0].tokens])
    return word_list

# Filters stop words in the word list
def filter_words_hindi(word_list):
    final_words = []
    for words in word_list:
        filtered_words = [word for word in words if word not in stop_words_hi]
        final_words.append(filtered_words)
    return final_words

# Removes punctuation
# Removes punctuation, including Hindi-specific punctuation (e.g., ।)
def remove_punct_hindi(word_list):
    hindi_punctuations = ['।', '!', '?', ',', '.', ':', ';', '-', '(', ')', '"', "'"]
    
    for sentence in word_list:
        sentence[:] = [word for word in sentence if word not in hindi_punctuations]
    return word_list


# Lemmatizes words (basic lowercasing example)
def lemmatize_hindi(word_list):
    for sentence in word_list:
        for i, word in enumerate(sentence):
            sentence[i] = word.lower()  # Basic lemmatization (lowercase)
    return word_list

# Translate final words to English using MarianMT
def translate_to_english(word_list):
    translated_words = []
    for sentence in word_list:
        translated_sentence = [translate(word) for word in sentence]
        translated_words.append(translated_sentence)
    return translated_words

# Main processing pipeline
def process_hindi_text(text):
    # Step 1: Convert text to sentence list
    sentences = convert_to_sentence_list_hindi(text)
    print("Sentences:", sentences)  # Debug: Print sentences
    
    # Step 2: Convert sentences to words
    word_list = convert_to_word_list_hindi(sentences)
    print("Words List:", word_list)  # Debug: Print word list
    
    # Step 3: Remove stop words
    filtered_words = filter_words_hindi(word_list)
    print("Filtered Words:", filtered_words)  # Debug: Print filtered words
    
    # Step 4: Remove punctuation
    clean_words = remove_punct_hindi(filtered_words)
    print("Clean Words:", clean_words)  # Debug: Print clean words
    
    # Step 5: Lemmatize words
    lemmatized_words = lemmatize_hindi(clean_words)
    print("Lemmatized Words:", lemmatized_words)  # Debug: Print lemmatized words
    
    # Step 6: Translate to English
    translated_words = translate_to_english(lemmatized_words)
    print("Translated Words:", translated_words)  # Debug: Print translated words
    
    return translated_words

# Test function
def test_hindi_to_english():
    text = "मैं स्कूल जा रहा हूँ"  # Sample Hindi text
    translated_words = process_hindi_text(text)
    print("Final Translated Words:", translated_words)

# Run the example directly
test_hindi_to_english()
"""

from googletrans import Translator

# Initialize the translator
translator = Translator()

def translate_with_google(text):
    translated = translator.translate(text, src='hi', dest='en')
    return translated.text

# Test with Google Translate
input_text = "मैं स्कूल जा रहा हूँ"
output_text = translate_with_google(input_text)
print(f"Input: {input_text}")
print(f"Google Translated Output: {output_text}")
