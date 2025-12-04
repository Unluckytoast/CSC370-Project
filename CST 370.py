

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
    "gato": [
        "cat (animal)",
        "car jack",
        "jack (car tool)",
       
    ],
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

    # Spanish sentence into a vector the AI can understand
    sent_embed = model.encode(spanish_sentence, convert_to_tensor=True)

    # Now do the same for all the possible English translations
    cand_list = candidates[target_word]
    cand_embeds = model.encode(cand_list, convert_to_tensor=True)

    # Compare how similar the sentence is to each possible translation
    cos_scores = util.cos_sim(sent_embed, cand_embeds)[0].cpu().numpy()

    # Context boost for 'gato' (car jack)
    if target_word == 'gato':
        context_words = ['coche', 'llanta', 'auto', 'carro', 'vehículo']
        sentence_lower = spanish_sentence.lower()
        if any(ctx in sentence_lower for ctx in context_words):
            for i, cand in enumerate(cand_list):
                cand_lower = cand.lower()
                if any(key in cand_lower for key in ['jack', 'car jack', 'car tool', 'lift']):
                    cos_scores[i] += 0.5  # Stronger boost for car jack meaning

    # Debug printout of all candidate scores
    print("\nCandidate translation scores:")
    for cand, score in zip(cand_list, cos_scores):
        print(f"  {cand}: {score:.3f}")

    # Sort them so the best matches come first, then grab the top k results
    ranked_idx = np.argsort(-cos_scores)
    topk = [(cand_list[i], float(cos_scores[i])) for i in ranked_idx[:k]]
    return topk

def interactive_loop():  
    print("Context Sensitive translation")
    print("Type a Spanish sentence containing the target word.")
    print("Example: 'Me senté en el banco del parque.' (I sat on the park bench.)")
    print("Type 'exit' to quit.\n")

    feedback_results = []
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

        # Always show both the full translation and the context-aware meaning for the ambiguous word
        print(f"\nFull English translation: {translation}")
        top_candidate = results[0][0] if results else None
        if top_candidate:
            print(f"Context-aware meaning for '{target_word}': {top_candidate}")

            # Ask user for the word to replace in the translation example: the cat used the cat in the car
            #top k translation candidate is car jack so we type cat in the car to get corect tanslation
            word_to_replace = input("\nWhich word in the English translation should be replaced with the context-aware meaning? (or press Enter to skip): ").strip()
            if word_to_replace:
                if word_to_replace in translation:
                    context_aware_translation = translation.replace(word_to_replace, top_candidate, 1)
                    print(f"Context-aware translation: {context_aware_translation}")
                else:
                    print("That word was not found in the translation. No replacement made.")

        # Ask user if the output was expected
        feedback = input("\nWas the output what you expected? (yes/no): ").strip().lower()
        if feedback == "yes":
            feedback_results.append(1)
            print("Thank you for your feedback!\n")
        elif feedback == "no":
            feedback_results.append(0)
            print("Sorry the output was not as expected.\n")
        else:
            print("Feedback not recognized.\n")

    # After loop ends, show accuracy graph
    if feedback_results:
        import matplotlib.pyplot as plt
        total = len(feedback_results)
        correct = sum(feedback_results)
        accuracy = correct / total if total > 0 else 0
        print(f"\nUser evaluation complete. Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        plt.bar(['Correct', 'Incorrect'], [correct, total-correct], color=['green', 'red'])
        plt.title('User Evaluation Accuracy')
        plt.ylabel('Count')
        plt.show()


if __name__ == "__main__":
    # Wake up model and run through a quick test makes everything faster after that
    print("Loading model and warming up (one-time)...")
    _ = model.encode("hola", convert_to_tensor=True)
    print("Ready!\n")
    interactive_loop() 
