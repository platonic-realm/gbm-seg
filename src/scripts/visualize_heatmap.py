import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import multiprocessing as mp


if __name__ == '__main__':
    # define a sphere about [0.5, 0.5, 0.5]
    sphere = np.load('/data/afatehi/gbm/morph_result.npy')

    shape = sphere[10:80, 1000:1200, 1000:1200]
    shape = shape.transpose((1, 2, 0))

    mask = shape != 0
    sphere = shape[mask]

    max = sphere.max()
    min = sphere.min()
    std = sphere.std()
    mean = sphere.mean()

    del sphere

    print(f"max={max}, min={min}")
    print(f"std={std}, mean={mean}")
    normalized = (shape - mean)/(std)

    # combine the color components
    cmap = cm.get_cmap('viridis')
    colors = cmap(normalized)

    def draw(_angle):
        ax.view_init(elev=15, azim=x)
        plt.savefig(f"/data/afatehi/figs/plot_{_angle}.png", bbox_inches='tight')

    # and plot everything
    x, y, z = np.indices((201, 201, 71))
    ax = plt.figure(figsize=(40, 40)).add_subplot(projection='3d')
    ax.voxels(x, y, z,
              shape,
              facecolors=colors,
              linewidth=1)
    ax.set(xlabel='r', ylabel='g', zlabel='b')
    ax.set_aspect('equal')

    plt.axis('off')

    processes = []
    for x in range(0, 90):
        process = mp.Process(target=draw, args=(x,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    processes = []
    for x in range(90, 180):
        process = mp.Process(target=draw, args=(x,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    processes = []
    for x in range(180, 270):
        process = mp.Process(target=draw, args=(x,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    processes = []
    for x in range(270, 360):
        process = mp.Process(target=draw, args=(x,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
