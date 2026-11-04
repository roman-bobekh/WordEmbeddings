# WordEmbeddings
Converting words to vectors using machine learning. A Keras model learns word embeddings by predicting words from their context window; `Embeddings.py` loads trained embeddings and supports closest-word queries, cosine similarity and vector arithmetic. Several trained models are in `models/`.

## Run
```python
from Embeddings import Embeddings
e = Embeddings("10kwikide_10epochs_64dim")
print(e.closestWords("king", 5))
```
Requires `numpy` (and `tensorflow` for training with `Model.py`).

## To-do
- exclude word itself from context
- check data selection
- better dataset (books?)
- better setting handling
- comments
- docs
