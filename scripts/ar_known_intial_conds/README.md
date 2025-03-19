### These are scripts Used for Learning autoregressive rollout on intial conditions seen in training

#### The Idea is that if we learn keep the intial conditions the same we can learn the physics and ideally be able to autoregressively learn past our training horizon

- The notebook is used for doing the entire process from splitting data to training model and evaluating the results
- train.py is used for training the model on already split data
- eval.py designed to be used after train.py to generate evaluation results of training model