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

def improved_segment_sentence(s):
    n = len(s)
    # Best score and best segmentation at each position
    dp = [(-float('inf'), None) for _ in range(n + 1)]
    dp[0] = (0, "")
    
    # Word length preference: longer words get higher scores
    def word_score(word):
        if word in dictionary:
            return len(word)**1.5  # Favor longer words more strongly
        return -len(word)  # Penalty for non-dictionary words
    
    for i in range(1, n + 1):
        for j in range(max(0, i-20), i):  # Look back up to 20 chars for efficiency
            word = s[j:i]
            score = word_score(word)
            
            if dp[j][0] + score > dp[i][0]:
                prev_text = dp[j][1]
                dp[i] = (dp[j][0] + score, (prev_text + " " + word).strip())
    
    return dp[n][1] if dp[n][1] else "No valid segmentation found"

def segment_with_chunking(s, chunk_size=15):
    """
    First divide the string into chunks, then segment each chunk separately.
    
    Args:
        s: Input string to segment
        chunk_size: Approximate size for each chunk
    
    Returns:
        Segmented string with spaces between words
    """
    # Step 1: Create overlapping chunks
    chunks = []
    overlap = 5  # Amount of overlap between chunks
    
    start = 0
    while start < len(s):
        end = min(start + chunk_size + overlap, len(s))
        chunks.append(s[start:end])
        start += chunk_size
    
    # Step 2: Segment each chunk
    segmented_chunks = []
    for chunk in chunks:
        segmented = segment_sentence(chunk)
        if segmented == "No valid segmentation found":
            # Fall back to character-by-character if no valid segmentation
            segmented = " ".join(chunk)
        segmented_chunks.append(segmented)
    
    # Step 3: Merge the segmented chunks, removing overlap
    result = segmented_chunks[0] if segmented_chunks else ""
    for i in range(1, len(segmented_chunks)):
        # Try to find a good joining point in the overlap region
        prev_chunk = chunks[i-1]
        curr_chunk = chunks[i]
        
        # Find overlap between chunks
        overlap_region = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
        
        # Find where the overlap appears in the segmented previous chunk
        prev_segmented = segmented_chunks[i-1]
        curr_segmented = segmented_chunks[i]
        
        # Simple merge - just concatenate with a space
        result += " " + curr_segmented
    
    return result

def multi_chunk_segment(s):
    """Try multiple chunk sizes and pick the best result"""
    best_score = -float('inf')
    best_segmentation = ""
    
    for chunk_size in [8, 10, 12, 15, 20]:
        segmented = segment_with_chunking(s, chunk_size)
        words = segmented.split()
        
        # Score based on number of dictionary words and their length
        score = sum(3 * (word in dictionary) + len(word) * 0.1 for word in words)
        
        if score > best_score:
            best_score = score
            best_segmentation = segmented
    
    return best_segmentation

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
    sentence_length = random.randint(2, 5)
    random_words = random.sample(list(dictionary), sentence_length)

    sentence = " ".join(random_words)
    concat_sentence = "".join(random_words)

    sentences.append([sentence, concat_sentence])


#after this, sentences holds tuples of strings (sentence, concat_sentence)
#ex. if sentence is "hello there" concat_sentence is "hellothere"
#will use this to test accuracy of our baseline heuristic segment_sentence

def heuristic_accuracy(numSentences=100, useless_parameter=80):

    accuracy = 0

    for i in range(numSentences):

        if random.randint(0,100) <= useless_parameter:
            segmented_sentence = segment_sentence(sentences[i][1])
        else:
            segmented_sentence = segment_sentence_mid_bad(sentences[i][1])

        print("input is " + sentences[i][1])
        print("output is " + segmented_sentence)

        if segmented_sentence == sentences[i][0]:
            accuracy = accuracy + 1
            print("Heuristic is correct!")
        else:
            print("Heuristic is wrong!")

        print(" ")

    accuracy = float(accuracy / numSentences)
    print("accuracy is " + str(accuracy))

    return accuracy

if __name__ == "__main__":
    # Test the heuristic accuracy with a specified number of sentences
    heuristic_accuracy(numSentences=5)
    