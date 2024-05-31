import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os
import nltk
from nltk.stem import WordNetLemmatizer

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = tf.keras.models.load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_interests = intents_json["intents"]
    for i in list_of_interests:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

def update_intents(user_input, correct_response):
    # Load intents
    with open("intents.json") as file:
        intents = json.load(file)
    # Check if the intent already exists
    for intent in intents["intents"]:
        if intent["tag"] == correct_response:
            intent["patterns"].append(user_input)
            intent["responses"].append(correct_response)
            break
    else:
        # If the intent does not exist, create a new one
        new_intent = {"tag": correct_response, "patterns": [user_input], "responses": [correct_response]}
        intents["intents"].append(new_intent)
    # Save updated intents
    with open("intents.json", "w") as file:
        json.dump(intents, file, indent=4)

def retrain_model():
    # Load the updated intents
    intents = json.loads(open("intents.json").read())
    
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))
    
    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))
    
    training = []
    
    # Prepare training data
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        output_row = [0] * len(classes)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])
    
    random.shuffle(training)
    training = np.array(training, dtype=object)  # Specify dtype as object to allow for mixed types
    
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
    
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save("chatbot_model.h5")


print("Zakviel is running")

while True:
    message = input("You: ").lower()
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
    
    feedback = input("Was this response helpful? (Yes/No): ").lower()
    if feedback == "no":
        correct_response = input("Please provide the correct response: ")
        update_intents(message, correct_response)
        retrain_model()
        print("Thank you! I've learned from your input.")
