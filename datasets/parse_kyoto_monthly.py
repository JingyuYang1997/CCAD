import pandas as pd
import numpy as np
import os
import random

src_path = "./Kyoto/Kyoto2016/"
years = ["2006","2007","2008","2009","2010","2011","2012","2013","2014","2015"]
target_parse_path = './Kyoto/kyoto_processed/monthly/parse'
target_subset_path = './Kyoto/kyoto_processed/monthly/subset'

bins = [1.1 ** i - 1 for i in range(233)]
bins[0] -= 0.01
bins[-1] += 0.01

for year in years:
    months = sorted(os.listdir(os.path.join(src_path, year)))
    for month in months:
        print("Parsing: Y{} M{}".format(year,month))
        ds_entries = []
        filenames = os.listdir(os.path.join(src_path, year, month))
        filenames = sorted(filenames)
        for filename in filenames:
            filepath = os.path.join(src_path, year, month, filename)
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()
                line_items = []
                for l in lines:
                    lsplit = l.split("\t")
                    timestamp = year + "_" + month + "_" + lsplit[-2]
                    protocol = lsplit[-1]
                    line_items.append(lsplit[:18] + [timestamp, protocol])

                ds_entries += line_items
        df_month = pd.DataFrame(ds_entries)

        for col in [0, 2, 3]:
            num_bins = len(bins)
            bin_names = ["c" + str(col).replace(" ", "") + str(idx) for idx in range(num_bins - 1)]
            df_month[col] = pd.cut(df_month[col].astype(float), bins=bins, labels=bin_names, ).astype(str)

        df_month.columns = [str(item) for item in range(20)]
        p_subset = 0.1
        random_state = int(year+month)
        random.seed(random_state)
        all_indexes = sorted(random.sample(list(range(df_month.shape[0])), int(p_subset * df_month.shape[0])))
        df_month_subset = df_month.iloc[all_indexes]
        df_month.to_parquet(os.path.join(target_parse_path,'{}_{}_full.parquet'.format(year,month)))
        df_month_subset.to_parquet(os.path.join(target_subset_path,'{}_{}_subset.parquet'.format(year,month)))
        del ds_entries
        del df_month
        del df_month_subset
