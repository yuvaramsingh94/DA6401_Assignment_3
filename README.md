# DA6401_Assignment_3

Author : V M Vijaya Yuvaram Singh (DA24S015)

## Report
[Wandb report](https://wandb.ai/yuvaramsingh/assignment_3/reports/Assignment-3-Yuvaram--VmlldzoxMjgxOTc1Ng)

## Github
[Link](https://github.com/yuvaramsingh94/DA6401_Assignment_3)

## Folder organization
```
DA6401_Assignment_2/
|   |── Part_a \\ codes for Part A question
|       |── inference.py \\ Code to make prediction
|       |── Config.py \\ Configuration of the best model
|       |── dataloader.py
|       |── RecursiveNetwork.py  \\ Module definition for all the recursive cells  
|       |── Seq2SeqModel.py \\ Torch lightning modules
|       |── sweep.py \\ Code to perform hyperparameter sweep
|       |── train.py \\ Code to train the model
|       |── utils.py
|   |── dataset
|   |── predictions_attention \\ Best attention model's prediction of attention 
|   |── predictions_vanilla \\ Best basic model's prediction of attention
|   |── weights
|   |── requirments.txt 
```

#### Hyper parameter tuning
```
python code/sweep.py
```
#### Training the best model
Edit the configuration file if required
```
python code/train.py
```

#### Inference the best model
Edit the configuration file if required
```
python code/inference.py
```



Note: I use COnfiguration file instead of command line to setup the parameters. Also on the assignment it is not stated to make code with commanline inputs.