"""
Flask Application - AlexNet Training and Prediction Service
"""
import os
import cv2
import json
import torch
# import tempfile
import subprocess
import numpy as np
import time
from skimage import io
import torchvision.models as models
from flask import Flask, render_template, request, redirect, url_for
# from time import sleep

os.environ['MKL_THREADING_LAYER'] = 'GNU'

app = Flask(__name__)
port = int(os.getenv("PORT"))
# port = 5001

# Checkpointing
resume = False

def train():
    global resume
    if resume:
        training_cmd = ['python', 'main.py', '-a', 'alexnet', '--resume', '.','--epochs', '1', '-b', '8', 'data']
    else:
        training_cmd = ['python', 'main.py', '-a', 'alexnet','--epochs', '1', '-b', '8', 'data']
    output = subprocess.run(training_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode('utf-8')
    os.system('rm /mnt/data/checkpoint.pth.tar')
    os.system('cp checkpoint.pth.tar /mnt/data/checkpoint.pth.tar')

    return render_template('index.html', text_1=output)
    

def model_classify(filepath):
    model = models.alexnet()
    model_ = torch.load('model_best.pth.tar', map_location={'cuda:0': 'cpu'})

    data = io.imread(filepath)
    data = cv2.resize(data, (224,224))
    data = np.reshape(data, (3,224,224))
    data = torch.Tensor(data).unsqueeze(0)
    
    model.load_state_dict(model_['state_dict'], strict=False)
    model.eval()

    output = model(data)
    _, predicted = torch.max(output.data, 1)
    with open('imagenet_labels.json') as labels_json:
        return json.load(labels_json)[predicted.item()]
    return None

def classify(req):
    uploaded_file = req.files['file']
    if uploaded_file.filename != '':
        uploaded_file.stream.seek(0)
        uploaded_file.save('data/classify/' + uploaded_file.filename)
        evaluation = model_classify(filepath = 'data/classify/' + uploaded_file.filename)
        os.system('rm data/classify/*')
        if evaluation:
            return render_template('index.html', text_2="The image is classified as " + evaluation)
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def main_page():
    try:
        if request.method == 'POST':
            if request.form['submit_btn'] == 'Train':
                return train()            
            elif request.form['submit_btn']== 'Classify':
                return classify(req=request)
        return render_template('index.html')
    except Exception as e:
        return " Service error: " + str(e)
    
if __name__ == '__main__':
    # Check if a checkpoint is present
    if os.path.exists('/mnt/data/checkpoint.pth.tar'):
        os.system('rm checkpoint.pth.tar')
        os.system('cp /mnt/data/checkpoint.pth.tar checkpoint.pth.tar')
        resume = True

    app.run(host='0.0.0.0', port=port, debug=True)
