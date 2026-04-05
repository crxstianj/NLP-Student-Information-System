# NLP Student Information System

## ¿Cómo funciona?

1. Se cargan las descripciones de los alumnos desde MongoDB
2. Cada descripción es **lematizada** (con spaCy en español) y se eliminan stopwords
3. Se construye una matriz **Bag of Words** con CountVectorizer
4. La consulta del usuario pasa por el mismo preprocesamiento y se compara contra la matriz usando **similitud coseno**
5. Se devuelven los 4 perfiles más similares, mostrando la foto del más cercano y guardando los resultados en `resultados.txt`

## Estructura esperada en MongoDB
```json
{
  "nombre_completo": "Juan Pérez",
  "descripcion": "Estudiante de sistemas interesado en IA y desarrollo web",
  "fotografia": "ruta/a/foto.jpg"
}
```

Base de datos: `alumnos_db` — Colección: `alumnos`

## Uso

Edita la variable `consulta` en `script.py` con el texto a buscar:
```python
consulta = "estudiante interesado en inteligencia artificial"
```

Luego ejecuta:
```bash
python script.py
```

Los resultados se guardan en `resultados.txt`.

## Dependencias
```bash
pip install pymongo scikit-learn spacy nltk pillow
python -m spacy download es_core_news_sm
```

En Python, descarga las stopwords de NLTK una sola vez:
```python
import nltk
nltk.download('stopwords')
```
