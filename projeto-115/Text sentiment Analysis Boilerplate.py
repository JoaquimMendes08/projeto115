import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["Câmera incrível. Vale o preço"]

# Tokenização
vocab_size = 10000
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(sentence)
# Crie um dicionário chamado word_index
word_index = tokenizer.word_index
texto = tokenizer.texts_to_sequences(sentence)
# Preenchendo a sequência
padtype = 'post'
max = 100
ttype = 'post'
padded = pad_sequences(texto, maxlen=max,padding=padtype, truncating=(ttype))
# Defina o modelo usando um arquivo .h5
model = tensorflow.keras.models.load_model("model.h5")
# Teste o modelo
result = model.predict(padded)
emocao = np.argmax(result, axis=1)
# Imprima o resultado
print(emocao)