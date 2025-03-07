import nltk
from nltk.corpus import words

# Download the word list if not already downloaded
nltk.download('words')

# Load the English words dataset
dictionary = words.words()

# Save to file
with open("words.txt", "w") as file:
    for word in dictionary:
        file.write(word + "\n")

print("Dictionary saved to words.txt")

def load_dictionary(file_path="words.txt"):
    with open(file_path, "r") as file:
        return set(file.read().splitlines())  # Convert to set for fast lookup
