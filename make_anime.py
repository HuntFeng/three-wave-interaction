import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter
from tqdm import tqdm

fig, ax =plt.subplots()
x = np.load("data/x.npy")
t = np.load("data/t.npy")

labels = ["pump", "seed", "plasma"]
lines = [ax.plot(x,np.zeros_like(x),label=labels[n])[0] for n in range(2)]

ax.set_xlabel("x ($\mu$m)")
ax.set_ylabel("normalized energy")

ax.legend()
ax.set_ylim([0,0.05])

x0 = 10
v = 298.14239699997194

def update(frame: int):
    for n, ln in enumerate(lines):
        a = np.load(f"data/a{n}/{frame}.npy")
        ln.set_ydata(np.sum(a**2, axis=1))
        # ln.set_ydata(np.abs(a)**2)
        ax.set_xlim([x0+v*t[frame]-100, x0+v*t[frame]+100])
    ax.set_title(f"$t={t[frame]:.6f}ps$")    
    return lines

# ani = FuncAnimation(fig, update, frames=np.arange(0,t.size,100))
# plt.show()

gifwriter = ImageMagickFileWriter()
with gifwriter.saving(fig, "anime1.gif", dpi=100):
    for frame in tqdm(range(0, 200*100, 100)):
        update(frame)
        gifwriter.grab_frame()
gifwriter.finish()