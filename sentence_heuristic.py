import nltk
from nltk.corpus import words

from english_dictionary_loader import load_dictionary

# Download the word list if not already downloaded
#nltk.download('words')

# Load the English words dataset
dictionary = load_dictionary()
#heuristic function as a baseline model to compare our sentence segmenting model against
#dynamic programming implementation using a dictionary of english words to segment sentences

def segment_sentence(s):
    n = len(s)
    dp = [None] * (n + 1)
    dp[0] = ""
    
    for i in range(1, n + 1):
        for j in range(i):
            word = s[j:i]
            if word in dictionary and dp[j] is not None:
                dp[i] = (dp[j] + " " + word).strip()
                break
    
    return dp[n] if dp[n] else "No valid segmentation found"

#different implemtation of segment sentence heuristic using greedy algo
def segment_sentence_greedy(s):
    segmented = []
    i = 0
    while i < len(s):
        for j in range(i + 1, len(s) + 1):
            word = s[i:j]
            if word in dictionary:
                segmented.append(word)
                i = j  # Move index forward to the next part of the string
                break
        else:
            # If no valid word is found, move forward by 1 character (bad behavior)
            segmented.append(s[i])
            i += 1

    return " ".join(segmented)

#alternate heuristic implementation
def segment_sentence_moderate(s):
    segmented = []
    i = 0
    while i < len(s):
        longest_word = None
        longest_end = i + 1

        # Look for the longest possible word
        for j in range(i + 1, len(s) + 1):
            word = s[i:j]
            if word in dictionary:
                longest_word = word
                longest_end = j  # Save position to continue from

        # If a valid word was found, use it
        if longest_word:
            segmented.append(longest_word)
            i = longest_end  # Move to the next part of the string
        else:
            # If no word is found, take the single letter (bad behavior)
            segmented.append(s[i])
            i += 1

    return " ".join(segmented)

#another alt heuristic
def segment_sentence_mid_bad(s):
    segmented = []
    i = 0
    while i < len(s):
        found = False

        # Look for the first word it finds, but skip words shorter than 3 letters if possible
        for j in range(i + 1, len(s) + 1):
            word = s[i:j]
            if word in dictionary and (len(word) > 2 or j == len(s)):  # Avoid tiny words unless no choice
                segmented.append(word)
                i = j  # Move forward
                found = True
                break
        
        # If no valid word found, take the first 2 characters (slightly better than 1)
        if not found:
            segmented.append(s[i:i+2])  # Grab 2 characters at a time (worse behavior)
            i += 2

    return " ".join(segmented)

#create random english sentences
import random

numSentences = 100
sentences = []
for i in range(numSentences):
    sentence_length = random.randint(2, 20)
    random_words = random.sample(list(dictionary), sentence_length)

    sentence = " ".join(random_words)
    concat_sentence = "".join(random_words)

    sentences.append([sentence, concat_sentence])


#after this, sentences holds tuples of strings (sentence, concat_sentence)
#ex. if sentence is "hello there" concat_sentence is "hellothere"
#will use this to test accuracy of our baseline heuristic segment_sentence

accuracy = 0

for i in range(numSentences):
    segmented_sentence = segment_sentence_moderate(sentences[i][1])
    if segmented_sentence == sentences[i][0]:
        accuracy = accuracy + 1
        #print(segmented_sentence + ":::::" + sentences[i][0])
    else:
        print(segmented_sentence + ":::::" + sentences[i][0])

accuracy = float(accuracy / numSentences)

print(accuracy)

