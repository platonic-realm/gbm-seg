from pathlib import Path
import tifffile


input_path = '/data/afatehi/gbm/experiments/random_serparation/datasets/ds_train/'


def main():
    input = Path(input_path)

    zeros = 0
    ones = 0

    for tiff_file in input.glob("*.tif"):
        image = tifffile.imread(tiff_file)
        label = image[:, 3, :, :]
        zeros += (label == 0).sum()
        ones += (label != 0).sum()

    sum = zeros + ones
    zeros_ratio = zeros / sum * 100
    ones_ratio = ones / sum * 100

    print(f"Zero ratio: {zeros_ratio}")
    print(f"Ones ratio: {ones_ratio}")


if __name__ == '__main__':
    main()
