# Distributed training & Suggestive resource allocation
This project is aimed to study the impact of distributed trained of Deep Learning models to understand if a predictive model can be designed to predict the epoch speed and time to accuracy. We selected image classification as the application and CNN models to conduct our experiments. 

We collected the training logs for around 75 configurations in which we varied model type, batch size, GPU type, number of GPUs, number of data loaders. Once the predictive model (also referred to as the recommender model) is trained, if the test error is low, we aim to make this model available to the end-user by hosting it over a Kubernetes cluster as a web application. 

Finally, this can become a prescriptive solution that can suggest a configuration to the user involving the least training cost before they consider investing in hardware.


## Environments
1. [Jupyter Notebooks](https://jupyter.org/)
2. [PyCharm](https://www.jetbrains.com/pycharm/)
2. [Spyder](https://www.spyder-ide.org/)
3. [Visual Studio Code](https://code.visualstudio.com/)
