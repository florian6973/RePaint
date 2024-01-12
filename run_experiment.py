from test import main, build_conf
import time
from matplotlib import pyplot as plt
import numpy as np

params = {
    "n": 2,
    "seed": 1,
    "total_it": 13,
    "jump_length": 1,
    "jump_n_sample": 1,
    "parallel": False,
}

processes = []
for total_it in [15, 16]:
    params["total_it"] = total_it
    conf = build_conf("image_size_inet", "inet64", **params)
    processes.append((conf, *main(conf)))
    print("Started process", processes[-1][1].pid)
    
maxes=  {}
still_running = True
while still_running:
    still_running = False
    for i, (conf, p, queue) in enumerate(processes):
        while not queue.empty():
            still_running = True                
            data = queue.get_nowait()
            if isinstance(data, tuple):
                if isinstance(data[0], str):
                    if data[0] == "times":       
                        data_arr = np.array(data[1])
                        x = np.arange(data_arr.shape[0])
                        maxes[i] = len(x)
                        plt.figure()
                        plt.plot(x, data_arr, marker='x')
                        plt.xlabel('Repaint step')
                        plt.ylabel("Iteration")
                        plt.tight_layout()
                        plt.savefig(conf["log_dir"] + "/times.png")
                        plt.close()
                        np.savetxt(conf["log_dir"] + "/times.csv", np.c_[x, data_arr], delimiter=",")
                    else:
                        print("p", i, data[0], data[1])
                        # if data[1] == "Sampling complete":
                        #     print("Done", i)
                else:
                    total = ""
                    if i in maxes:
                        total = maxes[i]
                    idx = data[0]
                    print("p", i, "idx", idx, total)        
        if p.is_alive():
            still_running = True
            
    time.sleep(1)

print("End")
