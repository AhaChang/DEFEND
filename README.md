# Fairness-aware Attributed Graph Anomaly Detection

## Requirements
This code requires the following:
- Python==3.8
- PyTorch==1.11.0+cu113
- torch_geometric==2.4.0
- pygod==1.0.0

## Usage
Take Reddit dataset as an example:

```
python train.py --data reddit --alpha 0.5 --gamma 1.5 --weight_corr 1e-10 --num_trial 10 --lr0 0.001
```

## Dataset
The Reddit and Twitter datasets are released by [FairGAD](https://openreview.net/forum?id=3cE6NKYy8x).