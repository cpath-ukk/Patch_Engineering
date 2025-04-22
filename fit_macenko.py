#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
from PIL import Image
import staintools

def parse_args():
    p = argparse.ArgumentParser(description="Fit Macenko normalizer using first CPU affinity")
    p.add_argument("--ref_image", required=True,
                   help="Path to reference image (e.g. H&E patch)")
    p.add_argument("--output_pickle", required=True,
                   help="Path where to save pickled normalizer")
    return p.parse_args()


def main():
    args = parse_args()

    # load and brightness-normalize reference image
    img = Image.open(args.ref_image)
    arr = np.array(img)
    arr_std = staintools.LuminosityStandardizer.standardize(arr)

    # fit Macenko normalizer
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(arr_std)

    # save pickle
    os.makedirs(os.path.dirname(args.output_pickle), exist_ok=True)
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(normalizer, f)

if __name__ == '__main__':
    main()
