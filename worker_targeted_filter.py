import os
import argparse
import random
import numpy as np
from PIL import Image
import staintools
import pickle
import json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_patches', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--filter_pairs', type=str, required=True)
    p.add_argument('--exclude_existing', action='store_true')
    p.add_argument('--patch_classes_json', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument("--mask_dir_name", default='mask_FINAL')
    p.add_argument("--image_dir_name", default='image')
    p.add_argument('--norm_pickle', required=True)
    p.add_argument('--stitch_masks', required=True)
    p.add_argument('--output_dir', required=True)
    return p.parse_args()


def stitch(img1, m1, img2, m2, sm):
    sm3 = np.expand_dims(sm, -1) * np.ones(3, np.uint8)
    return np.where(sm3==0, img1, img2), np.where(sm==0, m1, m2)


def main():
    args = parse_args()
    rnd = random.Random(args.seed)
    classes_map = json.load(open(args.patch_classes_json))
    filter_pairs = set(tuple(sorted(map(int, p.split('-')))) for p in args.filter_pairs.split(','))

    img_dir = os.path.join(args.data_root, args.image_dir_name)
    msk_dir = os.path.join(args.data_root, args.mask_dir_name)
    out_img = os.path.join(args.output_dir, args.image_dir_name); os.makedirs(out_img, exist_ok=True)
    out_msk = os.path.join(args.output_dir, args.mask_dir_name); os.makedirs(out_msk, exist_ok=True)

    mask_files = os.listdir(msk_dir)
    stitch_masks = os.listdir(args.stitch_masks)
    normalizer = pickle.load(open(args.norm_pickle, 'rb'))

    # Exclude patches that already have a combination we are looking for
    if args.exclude_existing:
        mask_files = [mask_file for mask_file in mask_files 
                      if not any(set(fp).issubset(classes_map[mask_file]) for fp in filter_pairs)
                      ]
    count = 0
    while count < args.n_patches:
        m1 = rnd.choice(mask_files)
        m2 = rnd.choice(mask_files)
    

        i1 = m1.replace('.png', '.jpg')
        i2 = m2.replace('.png', '.jpg')
        img1 = np.array(Image.open(os.path.join(img_dir, i1)))
        img2 = np.array(Image.open(os.path.join(img_dir, i2)))
        m1_arr = np.array(Image.open(os.path.join(msk_dir, m1)))
        m2_arr = np.array(Image.open(os.path.join(msk_dir, m2)))
        smf = rnd.choice(stitch_masks)
        sm = np.array(Image.open(os.path.join(args.stitch_masks, smf)))

        # initial stitch to check classes
        tmp_img, tmp_m = stitch(img1, m1_arr, img2, m2_arr, sm)
        classes_out = set(np.unique(tmp_m))
        if not any(set(fp).issubset(classes_out) for fp in filter_pairs):
            continue

        # brightness+stain normalization
        img1_std = staintools.LuminosityStandardizer.standardize(img1)
        img2_std = staintools.LuminosityStandardizer.standardize(img2)
        img1_m = normalizer.transform(img1_std)
        img2_m = normalizer.transform(img2_std)

        out_img_arr, out_msk_arr = stitch(img1_m, m1_arr, img2_m, m2_arr, sm)
        name = f"{i1[:-4]}_{i2[:-4]}_{smf[:-4]}"
        Image.fromarray(out_img_arr).save(os.path.join(out_img, name + '.jpg'))
        Image.fromarray(out_msk_arr).save(os.path.join(out_msk, name + '.png'))

        count += 1

if __name__ == '__main__':
    main()