# Importing necessary libraries
from os import path
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seq_len = 30

# Define model parameters
gan_args = ModelParameters(batch_size=24,
                           lr=5e-4,
                           noise_dim=32,
                           layers_dim=128,
                           latent_dim=24,
                           gamma=1)

train_args = TrainParameters(epochs=50,
                             sequence_length=seq_len,
                             number_sequences=3
                             )

# Read the data
obs_daily_df = pd.read_csv('original_redriver_inputs_nodate.csv', index_col=None)

streamflow_data = real_data_loading(obs_daily_df.values, seq_len)

cols = list(obs_daily_df.columns)

# Training the TimeGAN synthesizer
if path.exists('synthesizer_streamflow.pkl'):
    synth = TimeSeriesSynthesizer.load('synthesizer_streamflow.pkl')
else:
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
    synth.fit(obs_daily_df, train_args, num_cols=cols)
    synth.save('synthesizer_streamflow.pkl')

pass

# Generating new synthetic samples
synth_data = synth.sample(n_samples=len(streamflow_data))
print(synth_data[0].shape)

# Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
axes=axes.flatten()

time = list(range(1,25))
obs = np.random.randint(len(streamflow_data))

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': streamflow_data[obs][:, j],
                   'Synthetic': synth_data[obs].iloc[:, j]})
    df.plot(ax=axes[j],
            title = col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
plt.show()
