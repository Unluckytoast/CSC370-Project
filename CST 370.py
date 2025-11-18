

from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline

# Load AI model that understands both Spanish and English
# Using the MiniLM model because it's fast and doesn't eat up too much memory
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# Load Hugging Face translation model (Spanish to English)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Dictionary of Spanish words with their possible English meanings this is  just to show how it works
candidates = {
    "banco": ["bank (financial)", "bench (seat)", "shore", "counter"],
    "gato": ["cat", "jack (car tool)", "gadget"],
    "boca": ["mouth", "entrance", "opening"],
    "vela": ["candle", "sail", "vigil", "watch"],
    "cola": ["tail", "glue", "queue/line"],
    "planta": ["plant", "sole of foot", "floor/story of building"],
    "radio": ["radio", "radius", "radium"],
    "pico": ["beak", "peak/summit", "pickaxe", "a little bit"],
    "carta": ["letter", "menu", "card (playing card)", "chart"],
    "sobre": ["envelope", "on/over/about"],
    "derecho": ["right (direction)", "law", "straight", "right (entitlement)"],
    "corriente": ["current (electricity)", "current (water)", "common/ordinary"],
    "capital": ["capital (city)", "capital (money)", "capital (letter)"],
    "lima": ["lime (fruit)", "file (tool)", "Lima (city)"],
    "llama": ["flame", "llama (animal)", "he/she calls"],
    "cura": ["cure", "priest"],
    "mango": ["handle", "mango (fruit)"],
    "muñeca": ["wrist", "doll"],
}

def rank_translations(spanish_sentence: str, target_word: str, k: int = 3):
 
    target_word = target_word.lower()
    if target_word not in candidates:
        return []  # word not in dictionary

    #  Spanish sentence into a vector the AI can understand
    sent_embed = model.encode(spanish_sentence, convert_to_tensor=True)

    # Now do the same for all the possible English translations
    cand_list = candidates[target_word]
    cand_embeds = model.encode(cand_list, convert_to_tensor=True)

    # Compare how similar the sentence is to each possible translation
    # Higher score = better match for the context
    cos_scores = util.cos_sim(sent_embed, cand_embeds)[0].cpu().numpy()

    # Sort them so the best matches come first, then grab the top k results
    ranked_idx = np.argsort(-cos_scores)
    topk = [(cand_list[i], float(cos_scores[i])) for i in ranked_idx[:k]]
    return topk

def interactive_loop():  
    print("AI Smart translation")
    print("Type a Spanish sentence containing the target word.")
    print("Example: 'Me senté en el banco del parque.' (I sat on the park bench.)")
    print("Type 'exit' to quit.\n")

    while True:
        # Get the Spanish sentence from the user
        spanish_sentence = input("Spanish sentence: ").strip()
        if spanish_sentence.lower() in ("exit", "quit"):
            break 

        
        target_word = input("Target Spanish word in the sentence (exact form): ").strip().lower()
        if target_word == "":
            print("Please enter a target word.\n")
            continue 

   
        results = rank_translations(spanish_sentence, target_word, k=5)
        if not results:
            print(f"No candidate translations found for '{target_word}'.\n")
            continue  

       
        # Show them the results, best matches first
        print("\nRanked translations (score = similarity):")
        for cand, score in results:
            print(f" - {cand}    ({score:.3f})")
        print()

         # Translate the full sentence using Hugging Face model
        translation = translator(spanish_sentence)[0]['translation_text']
        print(f"\nFull English translation: {translation}")


if __name__ == "__main__":
    # Wake up model and run through a quick test makes everything faster after that
    print("Loading model and warming up (one-time)...")
    _ = model.encode("hola", convert_to_tensor=True)
    print("Ready!\n")
    interactive_loop() 
