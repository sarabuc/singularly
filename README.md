# Detect hate speech in tweets

Use Huggingface transformer libary and Tensorflow 2.0 to do text classification on detecting hate speech in tweets
and deploy trained model as REST API using Flask

Datasets can be downloaded [here](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)


## Setup

1.. Download the repository or use Git to clone it : 
```
git clone https://github.com/haizadtarik/hate-speech-detection.git
```
2. Install dependencies
```
pip install -r requirements
```
3. Train model
```
python train.py
```
4. Deploy API
```
python api.py
```


## Resources
1. [List of Huggingface pretrained model](https://huggingface.co/transformers/pretrained_models.html)




