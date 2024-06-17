import numpy as np, sklearn as sk, huggingface_hub as hf, matplotlib.pyplot as plt
from transformers import pipeline

classifier = pipeline("text-classification")     #.safetensors => to save all the paramaters, weights and biases
"""The above line creates 4 new files, a config file to save the configuration of the model like no. of droupout neurons, no. of hidden layers, etc., 
a .safetensors file to save all the paramaters, weights and biases, 
a tokenizer file to save the tokenizer used to tokenize the input text, parameters of the tokenizer like the maximum length of the input text, vocab_size etc.
a vocab file which contains parameters of the """