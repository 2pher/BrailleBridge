import openai
import os

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def fix_segmentation_errors(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you need higher accuracy
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects segmentation errors and typos in text."},
            {"role": "user", "content": f"Fix any segmentation errors or typos in this text: '{text}'. Only respond with the corrected text, nothing else and no quotations in the output."}
        ],
        temperature=0.2  # Lower temperature for more deterministic outputs
    )
    
    # Extract the corrected text from the response
    corrected_text = response.choices[0].message.content.strip()
    return corrected_text

if __name__ == "__main__":
    # Example usage
    segmented_text = "bai s a sheept hat sleeps all day"
    # segmented_text = "Matmatiks"  
    print(f"segmented text with typos: {segmented_text}\n")  # Should output "bai s a sheept hat sleeps all day"
    corrected_text = fix_segmentation_errors(segmented_text)
    print(f"Segmented text with typos fixed: {corrected_text}\n")  # Should output "Bai's a sheep that sleeps all day"