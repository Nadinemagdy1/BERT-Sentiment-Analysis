from transformers import pipeline

print("Loading BERT model...")

sentiment = pipeline("sentiment-analysis")

print("Model loaded!")

reviews = [
    "This product is amazing!",
    "The service was very bad.",
    "I like the design but it is expensive."
]

for review in reviews:
    result = sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {result[0]['label']}")
    print(f"Score: {result[0]['score']}\n")