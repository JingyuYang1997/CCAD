# Continual Contrastive Anomaly Detection Under Natural Data Distribution Shifts

To reproduce CCAD, you should

#### 1. Download the [Kyoto-2006+](http://www.takakura.com/Kyoto_data/new_data201704/) [$^{[1]}$](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7e5d5b3c53aec8ec833347001305d1b933c13a9a#page=32) Dataset

```
python ./datasets/download.py
```

#### 2. Process the Dataset

```
parse the txt files --> one-hot encoding ( refer to parse_kyoto_monthly.py and preprocess_onehot_monthly)
```

#### 3. Run CCAD

```
CCAD w/o rehearsal: python CCAD.py --gpu 0 
```

```
CCAD w rehearsal: python CCAD.py --gpu 0 --replay 
```

#### 4. Analyze the Results

```
python ./eval_results/parse_pkl.py
```



