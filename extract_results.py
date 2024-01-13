import numpy as np
import matplotlib.pyplot as plt


# times
data_times = r"experiments/evolutive_resampling/outputs/inet256_thick_middle/its_25_jl_5_js_5/logs/times.txt"
times = np.loadtxt(data_times)
print(times.shape)
plt.plot(times[:, 0], times[:, 1], label="ours")
plt.savefig("times-exp.png")