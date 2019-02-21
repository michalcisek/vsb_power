import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

train = pq.read_pandas('train.parquet').to_pandas()

metatrain = pd.read_csv('metadata_train.csv')

metatrain = metatrain.set_index(['id_measurement', 'phase'])

metatrain.head()

metatrain['target'].value_counts()

len(metatrain['signal_id'].unique())
metatrain.shape

metatrain['id_measurement'].value_counts()

plt.plot(train.iloc[:, 0:3])

type(train)
max_vals = train.apply(np.max, axis=1)

train_np = train.values

sample_size = 80000

max_vals = np.apply_along_axis(np.max, axis=1, arr=train_np)
min_vals = np.apply_along_axis(np.min, axis=1, arr=train_np)


