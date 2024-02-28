from fastapi import FastAPI, Body, HTTPException, status
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

app = FastAPI()

class Mensaje(BaseModel):
    texto: str

# Descargar recursos necesarios para NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Cargar los datos del archivo JSON
with open(r'intents.json', 'r', encoding='utf-8') as file:
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
    return {"categoria": categoria, "similitud": similitud_maxima}

@app.post('/categoriasRetorno/', tags=['cate'])
def retornarCategorias(mensaje: Mensaje):
    try:
        resultado = categorizar_mensaje(mensaje.texto)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
