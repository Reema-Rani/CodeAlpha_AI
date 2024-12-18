import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import numpy as np

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_md")

# Predefined FAQs (Questions and Answers)
faqs = {
    "What is the return policy?": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "How can I track my order?": "You can track your order using the tracking number provided in the confirmation email.",
    "What payment methods do you accept?": "We accept credit cards, debit cards, and PayPal.",
    "Do you offer international shipping?": "Yes, we offer international shipping to selected countries.",
    "How can I contact customer support?": "You can contact customer support via email at support@example.com or call us at 123-456-7890."
}

# Convert FAQs to embeddings
faq_questions = list(faqs.keys())
faq_answers = list(faqs.values())
faq_embeddings = [nlp(question).vector for question in faq_questions]
faq_embeddings = torch.tensor(faq_embeddings)

# Define a simple PyTorch similarity function
class SimilarityModel(nn.Module):
    def __init__(self):
        super(SimilarityModel, self).__init__()

    def forward(self, query_embedding, faq_embeddings):
        # Cosine similarity
        query_norm = F.normalize(query_embedding, p=2, dim=1)
        faq_norm = F.normalize(faq_embeddings, p=2, dim=1)
        similarity_scores = torch.mm(query_norm, faq_norm.T)
        return similarity_scores

# Instantiate the model
similarity_model = SimilarityModel()

# Function to process user query and find the most relevant FAQ
def get_response(user_query):
    query_embedding = nlp(user_query).vector.reshape(1, -1)
    query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

    with torch.no_grad():
        similarity_scores = similarity_model(query_embedding, faq_embeddings)

    # Find the index of the most similar FAQ
    max_score_index = torch.argmax(similarity_scores).item()
    best_match = faq_questions[max_score_index]
    response = faq_answers[max_score_index]
    
    return f"Q: {best_match}\nA: {response}"

# Chatbot interaction loop
def chatbot():
    print("Welcome to the FAQ chatbot! Ask your questions or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
