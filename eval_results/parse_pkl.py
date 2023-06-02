import pandas as pd
import os
import argparse
from ipdb import set_trace as debug

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument('-f', type=str)

    args = parser.parse_args()

    auroc_path = args.f+'_auroc.pkl'
    bwt_path = args.f+'_bwt.pkl'

    AUROCs = pd.read_pickle(auroc_path)
    BWTs = pd.read_pickle(bwt_path)


    AUROCs = [float(item) for item in AUROCs.values()]
    auroc = sum(AUROCs)/len(AUROCs)
    BWTs = [float(item.split('_')[-1]) for item in BWTs.keys()]
    bwt = sum(BWTs)/len(BWTs)

    print('{}\tAUROC: {:.4f}\tBWT: {:.4f}'.format(args.f,auroc,bwt))