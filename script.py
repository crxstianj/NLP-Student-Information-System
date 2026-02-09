from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.corpus import stopwords
import nltk
from PIL import Image
import os

# Cargar stopwords y lematizacion
#nltk.download('stopwords') #Una sola vez
nlp = spacy.load("es_core_news_sm")

# Conexion a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["alumnos_db"]
collection = db["alumnos"]

# Extraer descripciones de MongoDB
def cargar_descripciones():
    documentos = list(collection.find())
    return [doc["descripcion"] for doc in documentos], documentos

descripciones, documentos = cargar_descripciones()

# Lista de palabras vacias
stop_words = stopwords.words('spanish')

# Funcion para lematizar texto
def lematizar_texto(texto):
    doc = nlp(texto)
    return " ".join([token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct])

descripciones_lematizadas = [lematizar_texto(descripcion) for descripcion in descripciones]

# Crear modelo BoW
vectorizer = CountVectorizer(stop_words=stop_words)
matriz_bow = vectorizer.fit_transform(descripciones_lematizadas)

# Dimension del corpus
vocabulario = vectorizer.get_feature_names_out()
dimension_vocabulario = len(vocabulario)

# Funcion para buscar los mas similares
def buscar_similares(consulta):
    consulta_lematizada = lematizar_texto(consulta) 
    consulta_vectorizada = vectorizer.transform([consulta_lematizada])
    similitudes = cosine_similarity(consulta_vectorizada, matriz_bow)
    indices = similitudes.argsort()[0, -4:][::-1]
    resultados = [documentos[i] for i in indices]
    similitudes_top = similitudes[0][indices]  
    
    return resultados, similitudes_top

def mostrar_foto(ruta_foto):
    if os.path.exists(ruta_foto):
        imagen = Image.open(ruta_foto)
        imagen.show()
    else:
        print(f"No se encontro la imagen en: {ruta_foto}")

# Consulta
consulta = ""
resultados, similitudes = buscar_similares(consulta)

# Mostrar el mas similar
documento_mas_similar = resultados[0] 
similitud_maxima = similitudes[0] 

print(f"El mas similar es:")
print(f"Nombre: {documento_mas_similar['nombre_completo']}")
print(f"Similitud: {similitud_maxima}")

# Mostrar la foto del mas similar
foto_mas_similar = documento_mas_similar.get("fotografia")
if foto_mas_similar:
    mostrar_foto(foto_mas_similar)
else:
    print("No se encontro la foto")

# Guardar los resultados
with open('resultados.txt', 'w', encoding='utf-8') as f:
    f.write(f"Consulta: {consulta}\n")
    f.write("="*50 + "\n\n")
    for i, resultado in enumerate(resultados, 1):
        similitud = similitudes[i-1]  
        f.write(f"Resultado {i}:\n")
        f.write(f"Nombre: {resultado['nombre_completo']}\n")
        f.write(f"Descripción: {resultado['descripcion']}\n")
        f.write(f"Similitud: {similitud}\n")
        f.write("\n" + "-"*50 + "\n") 

print(f"Vocabulario guardado")
print(f"Resultados guardados")
print(f"Dimension del corpus: {dimension_vocabulario}")
