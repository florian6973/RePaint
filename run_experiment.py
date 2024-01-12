from test import main, build_conf
import time

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
    processes.append(main(conf))
    print("Started process", processes[-1][0].pid)

still_running = True
while still_running:
    still_running = False
    for i, (p, queue) in enumerate(processes):
        if p.is_alive():
            still_running = True
            while not queue.empty():
                data = queue.get_nowait()
                if isinstance(data, tuple):
                    if isinstance(data[0], str):
                        if data[0] == "times":
                            pass
                        else:
                            print("p", i, data[0], data[1])
                            if data[1] == "Sampling complete":
                                print("Done", i)
                    else:
                        idx = data[0]
                        print("p", i, "idx", idx)
    time.sleep(1)

print("End")
