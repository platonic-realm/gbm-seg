"""
Author: Arash Fatehi
Date:   07.11.2022
"""

# Local Imports
from src.models.unet3d import Unet3D
from src.test.visual import test_visualiztion_functinos

if __name__ == '__main__':
    model = Unet3D(3).to('cuda')
    test_visualiztion_functinos(model,
                                '/home/afatehi/gbm/data/GBM-Valid-DS',
                                (12, 256, 256),
                                (1, 16, 16),
                                4,
                                'cuda')
