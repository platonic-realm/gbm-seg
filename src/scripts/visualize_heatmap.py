import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import multiprocessing as mp


# define a sphere about [0.5, 0.5, 0.5]
sphere = np.load('/data/afatehi/gbm/result.npy')

print(f"max={sphere.max()}, min={sphere.min()}")
normalized = (sphere - sphere.min())/(10000)

print(sphere.std())

# combine the color components
cmap = cm.get_cmap('viridis')
colors = cmap(normalized)


def draw(_angle):
    ax.view_init(elev=15, azim=x)
    plt.savefig(f"/data/afatehi/figs/plot_{_angle}.png", bbox_inches='tight')


# and plot everything
x, y, z = np.indices((101, 101, 101))
ax = plt.figure(figsize=(20, 20)).add_subplot(projection='3d')
ax.voxels(x, y, z,
          sphere,
          facecolors=colors,
          linewidth=1)
ax.set(xlabel='r', ylabel='g', zlabel='b')
ax.set_aspect('equal')

processes = []
plt.axis('off')

for x in range(360):
    process = mp.Process(target=draw, args=(x,))
    processes.append(process)
    process.start()

for process in processes:
    process.join()
