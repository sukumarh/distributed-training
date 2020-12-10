# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
import os
from predict import predict

app = Flask(__name__)
#port = int(os.getenv("PORT"))
port = 5000

def get_results(request):
    data = {}
    data['num_of_paramters'] = int(request.form['num_of_paramters'])
    data['is_distributed'] = False
    data['num_workers_data_loader'] = int(request.form['num_workers_data_loader'])
    data['num_of_gpus'] = int(request.form['num_of_gpus'])
    data['batch_size'] = int(request.form['batch_size'])
    gpu = request.form['gpu']
    data['P40'] = 0
    data['P100'] = 0
    data['V100'] = 0
    data[gpu] = 1
    epoch_time, epoch_no = predict(data)
    text1 = "Time per epoch:" + str(epoch_time) + " seconds"
    text2 = "Number epochs to converge:" + str(epoch_no)
    return render_template('index.html', text1=text1, text2=text2)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    try:
        if request.method == 'POST':
            return get_results(request)
        return render_template('index.html')
    except Exception as e:
        return " Service error: " + str(e)

if __name__ == '__main__':
    
    app.run(host='localhost', port=port, debug=True)