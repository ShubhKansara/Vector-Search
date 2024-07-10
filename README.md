# Product Searching Using Vector Search

## Step 1: Install the Transformers Library

First, install the `sentence-transformers` library which is necessary for performing vector-based semantic searches.

```bash
!pip install sentence-transformers
```

## Step 2: Initialize the Library and Select Model
Load the pre-trained SentenceTransformer model to encode the product titles and search queries.

```python
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

## Step 3: Initialize Dataset
Prepare a sample dataset with product IDs, titles, and related keywords in a pandas DataFrame.

```python
data = [{'product_id': '1', 'title': 'Potato - બટાકા', 'related_keywords': 'Potato, Aalo, Alo, Aalu, Batata, Bateka, B, Bat, Bataka, A, P'}, 
        {'product_id': '2', 'title': 'Potato - બટાકા', 'related_keywords': 'Potato, Aalo, Alo, Aalu, Batata, Bateka, bataka'}, 
        {'product_id': '3', 'title': 'Potato - બટાકા', 'related_keywords': 'Potato, Aalo, Alo, Aalu, Batata, Bateka'}]

df = pd.DataFrame(data)
df
```


## Step 4: Prepare Dataset
Combine the product titles and related keywords, then encode them using the pre-trained model to obtain product embeddings.

```python
product_texts = df['title'] + ' ' + df['related_keywords']
product_embeddings = model.encode(product_texts.tolist())
```

## Step 5: Search for Products

Define a search query, encode it using the model, and find the products with the most similar embeddings. Display the products with similarity scores above a certain threshold.


```python
def search_products(search_query, threshold=0.5):
    # Encode the search query
    query_embedding = model.encode([search_query])[0]

    # Find the products with the most similar embeddings
    similar_products = []
    for i, product_embedding in enumerate(product_embeddings):
        similarity = util.cos_sim(query_embedding, product_embedding)[0][0].item()
        similar_products.append((similarity, i))

    # Sort the products by similarity
    similar_products.sort(key=lambda x: x[0], reverse=True)

    return similar_products

def print_similar_products(similar_products, threshold=0.5):
    print("_" * 200)
    print(f"Search Query: {search_query}")
    print("_" * 200)

    # Print the most similar products
    for similarity, i in similar_products:
        if similarity >= threshold:
            print(f"Product ID: {df['product_id'][i]}, Title: {df['title'][i]}, Similarity: {similarity:.4f}")

def visualize_similar_products(similar_products, top_n=10):
    top_similarities = [x[0] for x in similar_products[:top_n]]
    top_titles = [df['title'][x[1]] for x in similar_products[:top_n]]

    plt.figure(figsize=(10, 6))
    plt.barh(top_titles, top_similarities, color='skyblue')
    plt.xlabel('Similarity')
    plt.title('Top Matched Products')
    plt.gca().invert_yaxis()  # Highest similarity on top
    plt.show()

def search_and_visualize(search_query, threshold=0.5, top_n=10):
    # Search for similar products
    similar_products = search_products(search_query, threshold)
    
    # Print the similar products
    print_similar_products(similar_products, threshold)
    
    # Visualize the top matched products
    visualize_similar_products(similar_products, top_n)
```




## Step 6: Usage
Visualize the top matched products using a horizontal bar chart.


```python 
search_query = "sev tameta nu sak"
search_and_visualize(search_query)
```
