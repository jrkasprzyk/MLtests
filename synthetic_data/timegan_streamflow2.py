# Importing necessary libraries
from os import path
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seq_len = 12

# Define model parameters
gan_args = ModelParameters(batch_size=24,
                           lr=5e-4,
                           noise_dim=32,
                           layers_dim=128,
                           latent_dim=24,
                           gamma=1)

train_args = TrainParameters(epochs=5000,
                             sequence_length=seq_len,
                             number_sequences=1
                             )

# Read the data
#obs_daily_df = pd.read_csv('original_redriver_inputs_nodate.csv', index_col=None)
obs_monthly_af = pd.read_excel('oak_creek_monthly_af.xlsx', index_col=0)

values_array = obs_monthly_af["Volume (af)"].values
values_array = values_array.reshape(-1, 1)

streamflow_loader = real_data_loading(values_array, seq_len)

cols = ["Volume (af)"]

# Training the TimeGAN synthesizer
if path.exists('synthesizer_oakcreek.pkl'):
    synth = TimeSeriesSynthesizer.load('synthesizer_oakcreek.pkl')
else:
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
    synth.fit(obs_monthly_af, train_args, num_cols=cols)
    synth.save('synthesizer_oakcreek.pkl')

pass

# Generating new synthetic samples
num_samples = 100
synth_data = synth.sample(n_samples=num_samples)
print(synth_data[0].shape)

output_flows = pd.concat(synth_data, axis=0)
#output_flows.columns = range(num_samples)
output_flows.to_excel("./data/100years.xlsx")

# all_traces_df = pd.DataFrame()
#
# for i in range(num_samples):
#     #synth_data[i].to_excel(f"./data/output{i}.xlsx")
#     all_traces_df[i] = synth_data[i].values
#
# all_traces_df.to_excel("./data/all_traces.xlsx")


# # Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
# axes=axes.flatten()
#
# time = list(range(seq_len))
# obs = np.random.randint(len(streamflow_loader))
#
# for j, col in enumerate(cols):
#     df = pd.DataFrame({'Real': streamflow_loader[obs][:, j],
#                    'Synthetic': synth_data[obs].iloc[:, j]})
#     df.plot(ax=axes[j],
#             title = col,
#             secondary_y='Synthetic data', style=['-', '--'])
# fig.tight_layout()
# plt.show()
