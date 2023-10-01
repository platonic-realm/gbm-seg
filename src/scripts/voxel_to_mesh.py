import numpy as np
from skimage import measure
# import pyvista as pv
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

file_path = '/data/afatehi/gbm/experiments/DS2_Big_Mouse_37_12_Dice/results-infer/002-2920.pt_12128128_13232_6/CKM105.Nephrin.WGA.COLIV.20230507.lif - Series003.tif/morph_result.npy'

if __name__ == '__main__':
    tiff = np.load(file_path)
    mean = np.mean(tiff[tiff != 0])
    layer = tiff[0]
    layer[layer != 0] = mean/2
    tiff[0] = layer

    layer = tiff[tiff.shape[0]-1]
    layer[layer != 0] = mean/2
    tiff[tiff.shape[0]-1] = layer

    verts, faces, normals, values = measure.marching_cubes(volume=tiff,
                                                           level=0.1,
                                                           step_size=1.1,
                                                           allow_degenerate=False)

    np.save("verts.npy", verts)
    np.save("faces.npy", faces)
    np.save("values.npy", values)
    np.save("normals.npy", normals)
