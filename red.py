from fastapi import FastAPI, Body, HTTPException, status
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from keras.models import load_model
import numpy as np
import json

app = FastAPI()

class Mensaje(BaseModel):
    texto: str

# Descargar recursos necesarios para NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Cargar los datos del archivo JSON
with open(r'C:\fastAPI\intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Extraer las categorías y los mensajes
data = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        data.append({'MENSAJE': pattern, 'CATEGORIA': intent['tag']})

# Preprocesamiento del texto en los mensajes
stop_words = set(stopwords.words("spanish"))
lemmatizer = WordNetLemmatizer()

def preprocesar_texto(texto):
    tokens = nltk.word_tokenize(texto)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words and len(token) > 2]
    return ' '.join(tokens)

# Aplicar el preprocesamiento a cada mensaje
for item in data:
    item['MENSAJE_PREPROCESADO'] = preprocesar_texto(item['MENSAJE'])

# Crear un vectorizador TF-IDF y ajustarlo con los mensajes preprocesados
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([item['MENSAJE_PREPROCESADO'] for item in data])

# Función para categorizar mensajes
def categorizar_mensaje(mensaje):
    mensaje = str(mensaje)
    mensaje_preprocesado = preprocesar_texto(mensaje)
    mensaje_tfidf = vectorizer.transform([mensaje_preprocesado])
    similitudes = cosine_similarity(mensaje_tfidf, tfidf_matrix)
    similitud_maxima = similitudes.max()
    indice_maximo = similitudes.argmax()
    categoria = data[indice_maximo]['CATEGORIA'] if similitud_maxima >= 0.70 else 'otros'
    return categoria, similitud_maxima

# Cargar el modelo de red neuronal y los datos necesarios
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def classify_local(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

@app.post('/categoriasRetorno/', tags=['cate'])
def retornarCategorias(mensaje: Mensaje):
    try:
        # Clasificación con TF-IDF
        categoria_tfidf, similitud_tfidf = categorizar_mensaje(mensaje.texto)
        
        # Clasificación con el modelo de red neuronal
        resultados_red = classify_local(mensaje.texto)
        categoria_red, similitud_red = resultados_red[0] if resultados_red else ("otros", 0)
        
        # Convertir numpy.float32 a float para evitar problemas de serialización
        similitud_tfidf = float(similitud_tfidf)
        similitud_red = float(similitud_red)

        # Comparar resultados y devolver el de mayor similitud
        if similitud_tfidf > similitud_red:
            return {"categoria": categoria_tfidf, "similitud": similitud_tfidf}
        else:
            return {"categoria": categoria_red, "similitud": similitud_red}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
