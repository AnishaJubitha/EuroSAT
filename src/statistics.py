import pandas as pd
import numpy as np
import torch

# Calculating Mean and Std - for Normalisation
def mean_std_calc(loader, ch):
    
    sum = torch.zeros(ch, dtype=torch.float64)
    sq_sum = torch.zeros(ch, dtype=torch.float64)
    num_pixel = 0

    for images in loader:
        sum += images.sum(dim=[0, 2, 3])                # 0-Batch, 1-Channel, 2-Height, 3-Width
        sq_sum += (images ** 2).sum(dim=[0, 2, 3])

        num_pixel += images.size(0) * images.size(2) * images.size(3)

    new_mean = sum / num_pixel

    new_std = torch.sqrt(sq_sum / num_pixel - new_mean ** 2)

    print('MEAN : ', new_mean)
    print('Standard Deviation : ', new_std)

    MEAN = new_mean.tolist()
    STD = new_std.tolist()

    return MEAN, STD


def get_band_statistics(band_values):
    
    band_statistics = []

    for band_idx, arrays in enumerate(band_values):
        values = np.concatenate(arrays)

        band_statistics.append({
            'band_index' : band_idx + 1,
            'mean' : values.mean(),
            'std' : values.std(),
            'min' : values.min(),
            'p1' : np.percentile(values, 1),
            'median' : np.percentile(values, 50),
            'p99' : np.percentile(values, 99),
            'max' : values.max()
        })

    band_statistics = pd.DataFrame(band_statistics)

    return band_statistics