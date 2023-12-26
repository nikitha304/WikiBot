from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer, T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import os
import json
import time
from scipy.interpolate import make_interp_spline
app = Flask(__name__, static_folder='static')

model_name = 'facebook/blenderbot_small-90M'
tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)
respTimes = []
lengthVsTime = []
chitchat_query_probs_list = []
topic_probs_list = []
last_chitchat_query_prob = None
last_topic_probs = None

def chat_with_blenderbot(input_text):
    inputs = tokenizer([input_text], return_tensors='pt')
    reply_ids = model.generate(**inputs)
    resp = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    respnew = '. '.join(list(map(lambda x: x.strip().capitalize(), resp.split('.'))))
    return respnew

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text

def retrieve_and_rank_documents(specific_query, topics, data, k=5):
    all_docs = pd.DataFrame()

    for topic in topics:
        topic_data = data[data['topic'] == topic]

        if not specific_query.strip():
            top_docs = topic_data.head(k)
        else:
            vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
            try:
                tfidf_matrix = vectorizer.fit_transform(topic_data['summary'])
            except ValueError:
                print("Empty vocabulary; switching to default top documents.")
                top_docs = topic_data.head(k)
                all_docs = pd.concat([all_docs, top_docs])
                continue

            query_vector = vectorizer.transform([specific_query])
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            top_indices = cosine_similarities.argsort()[-k:][::-1]
            top_docs = topic_data.iloc[top_indices]

        all_docs = pd.concat([all_docs, top_docs])

    return all_docs

def generate_summary(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # print('summary',summary)
    summarynew = '. '.join(list(map(lambda x: x.strip().capitalize(), summary.split('.'))))
    # print(summarynew)
    return summarynew

    return summary

def classify_nature(query, classifier, vectorizer):
    query_vec = vectorizer.transform([query])
    nature = classifier.predict(query_vec)[0]
    nature_probs = classifier.predict_proba(query_vec)[0]
    last_chitchat_query_prob = {'ChitChat': nature_probs[0], 'Query': nature_probs[1]}
    return nature, nature_probs, last_chitchat_query_prob

def classify_query(query, classifier, vectorizer, threshold=0.30):
    query_processed = preprocess_text(query)
    query_vec = vectorizer.transform([query_processed])
    probabilities = classifier.predict_proba(query_vec)[0]
    last_topic_probs = {classifier.classes_[i]: prob for i, prob in enumerate(probabilities) if prob > threshold}
    topics = [classifier.classes_[i] for i, prob in enumerate(probabilities) if prob > threshold]
    return topics, probabilities, last_topic_probs

def plot_chitchat_vs_query_probabilities(probabilities):
    if not probabilities:
      x = 'Need a few more queries to display the graph!'
      y = 0
      plt.title("ChitChat vs Query Probabilities")
      plt.scatter(x, y, color="red", marker='o')
      plt.plot(x, y, color="blue")
      plt.xlabel("User input")
      plt.ylabel("Probability")
      buffer = BytesIO()
      plt.savefig(buffer, format='png')
      plt.close()
      buffer.seek(0)
      chitchat_vs_query = base64.b64encode(buffer.getvalue()).decode('utf-8')
      return chitchat_vs_query

    chitchat_probs = [prob['ChitChat'] for prob in probabilities]
    query_probs = [prob['Query'] for prob in probabilities]
    indices = np.arange(len(chitchat_probs))

    spline_degree = min(3, len(chitchat_probs) - 1)

    xnew = np.linspace(indices.min(), indices.max(), 300)
    spl_chitchat = make_interp_spline(indices, chitchat_probs, k=spline_degree)
    smooth_chitchat = spl_chitchat(xnew)
    spl_query = make_interp_spline(indices, query_probs, k=spline_degree)
    smooth_query = spl_query(xnew)

    plt.figure(figsize=(10, 6))
    plt.plot(xnew, smooth_chitchat, label='ChitChat', color='blue')
    plt.plot(xnew, smooth_query, label='Query', color='orange')
    plt.xlabel('User input')
    plt.ylabel('Probability')
    plt.title('ChitChat vs Query Probabilities')
    plt.xticks(indices, [f'input {i+1}' for i in range(len(chitchat_probs))])
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    chitchat_vs_query = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return chitchat_vs_query

def plot_topic_classification_probabilities(probabilities):
    if not probabilities:
      x = 'Need a few more queries to display the graph!'
      y = 0
      plt.title("Topic Classification Probabilities")
      plt.scatter(x, y, color="red", marker='o')
      plt.plot(x, y, color="blue")
      plt.xlabel("User input")
      plt.ylabel("Probability")
      buffer = BytesIO()
      plt.savefig(buffer, format='png')
      plt.close()
      buffer.seek(0)
      topic_prob = base64.b64encode(buffer.getvalue()).decode('utf-8')
      return topic_prob

    topics = set().union(*[list(prob.keys()) for prob in probabilities])
    topic_values = {topic: [] for topic in topics}

    for prob in probabilities:
        for topic in topics:
            topic_values[topic].append(prob.get(topic, 0))

    indices = np.arange(len(probabilities))
    xnew = np.linspace(indices.min(), indices.max(), 300)
    spline_degree = min(3, len(probabilities) - 1)

    plt.figure(figsize=(10, 6))
    for topic, values in topic_values.items():
        spl_topic = make_interp_spline(indices, values, k=spline_degree)
        smooth_topic = spl_topic(xnew)
        plt.plot(xnew, smooth_topic, label=topic)

    plt.xlabel('User Input')
    plt.ylabel('Probability')
    plt.title('Topic Classification Probabilities')
    plt.xticks(indices, [f'input {i+1}' for i in range(len(probabilities))])
    plt.legend()
    # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    topic_prob = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return topic_prob

binary_classifier = load('multinomial_nb_model.joblib')
binary_vectorizer = load('tfidf_vectorizer.joblib')
topic_classifier = load('classifier_model.joblib')
topic_vectorizer = load('vectorizer.joblib')

data = pd.read_pickle('dataNew.pkl')

@app.route('/')
def chatbot_interface():
    return render_template('index11working.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    # global last_chitchat_query_prob, last_topic_probs
    user_input = request.form['user_input']
    start_time = time.time()
    resp_time = 0
    selected_topics_json = request.form.get('selected_topics', '[]')
    selected_topics = json.loads(selected_topics_json)
    print(selected_topics)
    nature, nature_probs, last_chitchat_query_prob = classify_nature(user_input, binary_classifier, binary_vectorizer)
    chitchat_prob, query_prob = nature_probs

    chitchat_query_probs_list.append(last_chitchat_query_prob)
    topic_probs = np.array([])
    response = ""

    if nature == 'ChitChat':
        response = chat_with_blenderbot(user_input)
        resp_time = time.time() - start_time
        respTimes.append(resp_time)
        lengthVsTime.append((len(user_input), resp_time))
        print(selected_topics)
    else:
        try:
            topics, topic_probs, last_topic_probs = classify_query(user_input, topic_classifier, topic_vectorizer, 0.3)
            print(len(selected_topics))
            if(len(selected_topics)>=1):
                topics=selected_topics
            print(topics)
            if topics:
                specific_query = preprocess_text(user_input)
                top_documents = retrieve_and_rank_documents(specific_query, topics, data)
                print(top_documents)
                full_text = ' '.join(top_documents['summary'])
                summary = generate_summary(full_text)
                response = f"{summary}"
                resp_time = time.time() - start_time
            else:
                print("No relevant topics found")
                response = chat_with_blenderbot(user_input)
                resp_time = time.time() - start_time
            if last_topic_probs:
              print("Topic Classification Probabilities:", last_topic_probs)
              topic_probs_list.append(last_topic_probs)
        except Exception as e:
            
            print(f"An error occurred: {e}")
            response = chat_with_blenderbot(user_input)
            resp_time = time.time() - start_time
        respTimes.append(resp_time)
        lengthVsTime.append((len(user_input), resp_time))


    if len(respTimes) < 4:
      x = 'Need at least 4 queries to display a graph!'
      y = 0
      plt.title("Query Response Times")
      plt.scatter(x, y, color="red", marker='o')
      plt.plot(x, y, color="blue")
      plt.xlabel("Queries")
      plt.ylabel("Time")
      buffer = BytesIO()
      plt.savefig(buffer, format='png')
      plt.close()
      buffer.seek(0)
      query_resp_time = base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
      y = np.array(respTimes)
      x = np.arange(len(y))

      x_smooth = np.linspace(min(x), max(x), 300)
      spl = make_interp_spline(x, y, k=3)
      y_smooth = spl(x_smooth)

      plt.title("Query Response Times")
      plt.scatter(x, y, color="red", marker='o')
      plt.plot(x_smooth, y_smooth, color="blue")
      plt.xlabel("Queries")
      plt.ylabel("Time")
      buffer = BytesIO()
      plt.savefig(buffer, format='png')
      plt.close()
      buffer.seek(0)
      query_resp_time = base64.b64encode(buffer.getvalue()).decode('utf-8')

    if len(lengthVsTime) < 4:
      x = 'Need at least 4 queries to display a graph!'
      y = 0
      plt.title("Length VS Time")
      plt.scatter(x, y, color="red", marker='o')
      plt.plot(x, y, color="blue")
      plt.xlabel("Time")
      plt.ylabel("Length")
      buffer = BytesIO()
      plt.savefig(buffer, format='png')
      plt.close()
      buffer.seek(0)
      length_vs_time = base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
      y, x = zip(*lengthVsTime)

      x = np.array(x)
      y = np.array(y)
      sorted_indices = np.argsort(x)
      x = x[sorted_indices]
      y = y[sorted_indices]

      x_smooth = np.linspace(min(x), max(x), 300)
      spl = make_interp_spline(x, y, k=3)
      y_smooth = spl(x_smooth)

      plt.title("Length Of Query VS Time")
      plt.scatter(x, y, color="red", marker='o')
      plt.plot(x_smooth, y_smooth, color="blue")
      plt.xlabel("Time")
      plt.ylabel("Length")
      buffer = BytesIO()
      plt.savefig(buffer, format='png')
      plt.close()
      buffer.seek(0)
      length_vs_time = base64.b64encode(buffer.getvalue()).decode('utf-8')


    chitchat_query_plot_base64 = plot_chitchat_vs_query_probabilities(chitchat_query_probs_list)
    topic_plot_base64 = plot_topic_classification_probabilities(topic_probs_list)
    response_data = {
        'bot_response': response,
        'query_resp_plot' : query_resp_time,
        'length_time_plot': length_vs_time,
        'chitchat_query_plot': chitchat_query_plot_base64,
        'topic_plot':topic_plot_base64
    }
    print(user_input)
    print(response)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
