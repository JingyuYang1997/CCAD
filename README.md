# Continual Contrastive Anomaly Detection Under Natural Data Distribution Shifts[paper](https://ieeexplore.ieee.org/document/10208545)
This is a conference paper published on CACRE 2023.

To reproduce CCAD, you should

#### 1. Download the [Kyoto-2006+](http://www.takakura.com/Kyoto_data/new_data201704/) [ref](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7e5d5b3c53aec8ec833347001305d1b933c13a9a#page=32) Dataset

```
python ./datasets/download.py
```

#### 2. Process the Dataset (Refer to [Anoshift](https://proceedings.neurips.cc/paper_files/paper/2022/file/d3bcbcb2a7b0b4716bf24ce4b2ea8d60-Paper-Datasets_and_Benchmarks.pdf))

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



