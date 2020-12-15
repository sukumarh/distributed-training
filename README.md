# Distributed training & Suggestive resource allocation
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is aimed to study the impact of distributed trained of Deep Learning models to understand if a predictive model can be designed to predict the epoch speed and time to accuracy. We selected image classification as the application and CNN models to conduct our experiments. 

We collected the training logs for around 75 configurations in which we varied model type, batch size, GPU type, number of GPUs, number of data loaders. Once the predictive model (also referred to as the recommender model) is trained, if the test error is low, we aim to make this model available to the end-user by hosting it over a Kubernetes cluster as a web application. 

Finally, this can become a prescriptive solution that can suggest a configuration to the user involving the least training cost before they consider investing in hardware.

## Trainers
#### Single GPU Trainer
```bash
trainer_pytorch.py [-h] [-b BATCH_SIZE] [-c CONFIGURATIONS]
                        [--configuration-file CONFIGURATION_FILE] [-d DATA]
                        [--dataset DATASET] [-e EPOCHS] 
                        [-lr LEARNING_RATE] [-m MODEL_NAME]                          
                        [-w NUM_WORKERS] [-s SAVE_LOCATION]
```

#### Distributed Trainer (Multi-GPU)
```bash
distributed_trainer.py [-h] [-b BATCH_SIZE] [-c CONFIGURATIONS]
                            [--configuration-file CONFIGURATION_FILE]
                            [-d DATA] [--dataset DATASET]
                            [--distribute-data] [-da DISTRIBUTED_ADDRESS]
                            [-dp DISTRIBUTED_PORT]
                            [--distributed-backend DISTRIBUTED_BACKEND]
                            [-e EPOCHS] [-g GLOO_FILE] [-lr LEARNING_RATE]
                            [-m MODEL_NAME] [--num-nodes NUM_NODES]
                            [--num-gpus NUM_GPUS] [-w NUM_WORKERS]
                            [-s SAVE_LOCATION]
```

## Evaluations
The following graph shows the epoch timings for various configurations. In this experiment, each GPU trained on the entire dataset, leading to an increase in the epoch time with a larger decrease in the number of epochs required to reach a certain accuracy.

![Evaluations](/Resources/avg_epoch_time.jpg)

### Recommender model
##### Time per epoch (in seconds) 
| MAE  | RMSE |
|------|------|
| 1.84 | 4.60 |

##### Accuracy for an epoch
| MAE   | RMSE |
|-------|------|
| 0.047 | 0.10 |

## Frameworks & Libraries
1. [PyTorch](https://pytorch.org/)
    - [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
2. [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)

## Environments
1. [Jupyter Notebooks](https://jupyter.org/)
2. [PyCharm](https://www.jetbrains.com/pycharm/)
2. [Spyder](https://www.spyder-ide.org/)
3. [Visual Studio Code](https://code.visualstudio.com/)
