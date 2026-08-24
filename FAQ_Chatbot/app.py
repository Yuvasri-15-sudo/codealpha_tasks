from flask import Flask, render_template, request, jsonify
import json
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)


# =====================================================
# NLTK STOPWORDS
# =====================================================

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


stop_words = set(stopwords.words("english"))


# =====================================================
# LOAD FAQ DATA
# =====================================================

with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)


# =====================================================
# TEXT PREPROCESSING
# =====================================================

def preprocess_text(text):

    text = text.lower()

    # Common e-commerce phrase normalization
    replacements = {
        "send back": "return",
        "give back": "return",
        "take back": "return",
        "ship back": "return",
        "go back": "return",

        "money back": "refund",
        "get my money": "refund",

        "package": "order",
        "parcel": "order",
        "shipment": "order",

        "broke": "damaged",
        "broken": "damaged",

        "login password": "password",
        "sign in password": "password"
    }

    for phrase, replacement in replacements.items():
        text = text.replace(phrase, replacement)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =====================================================
# PREPARE FAQ QUESTIONS
# =====================================================

faq_questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]


# =====================================================
# TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True
)

faq_vectors = vectorizer.fit_transform(faq_questions)


# =====================================================
# SPECIAL INTENT KEYWORDS
# =====================================================

intent_keywords = {

    "return": [
        "return",
        "send back",
        "give back",
        "take back",
        "ship back",
        "exchange"
    ],

    "refund": [
        "refund",
        "money back",
        "money returned",
        "get my money"
    ],

    "late_order": [
        "late",
        "delay"
        "overdue"
        "not arrived"
        "hasn't arrived"
        "has not arrived"
        "delivery is late"
        "order is late"
        "order delayed"
    ],

    "cancel": [
        "cancel",
        "cancellation"
    ],

    "track": [
        "track",
        "tracking",
        "where is my order",
        "order status"
    ],

    "damaged": [
        "damaged",
        "broken",
        "defective"
    ],

    "missing": [
        "missing",
        "not received",
        "didn't receive",
        "did not receive"
    ]
}


# =====================================================
# FIND KEYWORD INTENT
# =====================================================

def detect_intent(question):

    # Normalize the question first
    question = question.lower()

    # Convert common e-commerce terms
    question = question.replace("package", "order")
    question = question.replace("parcel", "order")
    question = question.replace("shipment", "order")

    # Check tracking intent
    tracking_phrases = [
        "track",
        "tracking",
        "where is my order",
        "where is order",
        "order status",
        "check my order",
        "locate my order"
    ]

    for phrase in tracking_phrases:
        if phrase in question:
            return "track"

    # Check other intents
    for intent, keywords in intent_keywords.items():

        if intent == "track":
            continue

        for keyword in keywords:

            if keyword in question:
                return intent

    return None


# =====================================================
# FIND FAQ BY INTENT
# =====================================================

def find_intent_match(user_question):

    intent = detect_intent(user_question)

    if intent is None:
        return None

    intent_faq_mapping = {

        "return": [
            "How do I return a product?",
            "What is your return policy?",
            "How many days do I have to return an item?",
            "Can I exchange a product instead of returning it?"
        ],

        "refund": [
            "When will I receive my refund?",
            "Where will my refund be credited?"
        ],

        "late order": [
            "What should I do if my order is delayed?",
            "What should I do if my order is late?"
        ],

        "cancel": [
            "Can I cancel my order?"
        ],

        "track": [
            "How can I track my order?"
        ],

        "damaged": [
            "What should I do if I receive a damaged product?",
            "What should I do if I receive an incorrect or defective product?"
        ],

        "missing": [
            "What if an item is missing from my order?"
        ]
    }

    possible_questions = intent_faq_mapping.get(intent, [])

    for faq in faqs:

        if faq["question"] in possible_questions:

            return {
                "answer": faq["answer"],
                "confidence": 95
            }

    return None


# =====================================================
# BEST FAQ MATCH
# =====================================================

def get_best_answer(user_question):

    # First check for strong intent
    intent_result = find_intent_match(user_question)

    if intent_result:
        return intent_result


    # Preprocess question
    processed_question = preprocess_text(user_question)


    # Empty input after preprocessing
    if not processed_question:

        return {
            "answer": "Please enter a meaningful question.",
            "confidence": 0
        }


    # Convert user question to TF-IDF
    user_vector = vectorizer.transform(
        [processed_question]
    )


    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )


    # Find best match
    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[
        0
    ][best_match_index]


    # Convert to percentage
    confidence = round(
        float(best_score) * 100,
        2
    )


    # Confidence threshold
    threshold = 0.25


    if best_score < threshold:

        return {
            "answer": (
                "I'm sorry, I couldn't find a suitable "
                "answer to your question. Please try asking "
                "in a different way or contact our customer "
                "support team."
            ),
            "confidence": confidence
        }


    # Get answer
    best_answer = faqs[
        best_match_index
    ]["answer"]


    return {
        "answer": best_answer,
        "confidence": confidence
    }


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")


# =====================================================
# CHAT API
# =====================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_question = data.get(
        "question",
        ""
    ).strip()


    if not user_question:

        return jsonify({
            "answer": "Please enter a question.",
            "confidence": 0
        })


    result = get_best_answer(
        user_question
    )


    return jsonify(result)


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )