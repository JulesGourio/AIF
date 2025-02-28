import gradio as gr
from PIL import Image
import requests
import io


import io
import requests
from PIL import Image
import numpy as np

def recognize_digit(image):
    image = image['composite']
    image = Image.fromarray(image.astype('uint8'))
    image = image.resize((28, 28))
    image = image.convert('L')
    images_binary = io.BytesIO()
    image.save(images_binary, format='PNG')
    reponse = requests.post('http://127.0.0.1:5000/predict', data=images_binary.getvalue())
    return reponse.json()['prediction']


if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);