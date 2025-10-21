import numpy as np
import matplotlib.pyplot as plt
import time

# --- simulation variables (you can replace these with your actual inputs)
history = {"t": [], "spd": [], "tilt": [], "rpm": [], "aos": []}
lines = {}

plt.ion()  # turn on interactive mode
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs = axs.flatten()

labels = ["spd", "tilt", "rpm", "aos"]
for i, label in enumerate(labels):
    lines[label], = axs[i].plot([], [], label=label)
    axs[i].set_title(label)
    axs[i].set_xlim(0, 10)
    axs[i].set_ylim(-10, 100)
    axs[i].legend()

start_time = time.time()

for k in range(500):  # simulate 500 updates
    t = time.time() - start_time

    # fake data for testing
    history["t"].append(t)
    history["spd"].append(20 + 5*np.sin(0.2*t))
    history["tilt"].append(30 + 10*np.sin(0.5*t))
    history["rpm"].append(2000 + 500*np.sin(0.3*t))
    history["aos"].append(5*np.sin(0.1*t))

    # keep last 10 seconds of data
    if len(history["t"]) > 200:
        for key in history:
            history[key] = history[key][-200:]

    # update plots
    for label in labels:
        lines[label].set_data(history["t"], history[label])
        axs[labels.index(label)].set_xlim(max(0, t-10), t)

    plt.pause(0.05)

plt.ioff()
plt.show()