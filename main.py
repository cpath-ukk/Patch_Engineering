#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import yaml
import numpy as np
import json
from multiprocessing import Process, Queue
from PIL import Image


# -----------------------------------------------------------------------------
# Parse CLI for config path
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Launcher for patch‑stitching via YAML config")
    p.add_argument("--config", required=True,
                   help="Path to YAML configuration file")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Load YAML config
# -----------------------------------------------------------------------------

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Fit or load Macenko normalizer pinned to first CPU
# -----------------------------------------------------------------------------

def fit_normalizer(ref_image, pickle_path, cpus):
    if os.path.exists(pickle_path):
        return
    print(f"Fitting macenko normalizer and storing it at {pickle_path}")
    base = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(base, 'fit_macenko.py')
    first_cpu = str(cpus[0])
    cmd = [
        'taskset', '-c', first_cpu,
        sys.executable, script,
        '--ref_image', ref_image,
        '--output_pickle', pickle_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Done with fitting macenko normalizer")

# -----------------------------------------------------------------------------
# Read combos from 2D matrix in config
# -----------------------------------------------------------------------------

def read_combos(cfg):
    matrix = cfg['matrix']
    classes = cfg.get('classes', list(range(len(matrix))))
    combos = []
    for i_idx, row in enumerate(matrix):
        for j_idx, n in enumerate(row):
            if j_idx > i_idx and isinstance(n, (int,float)) and n > 0:
                combos.append((classes[i_idx], classes[j_idx], int(n)))
    return combos

# -----------------------------------------------------------------------------
# Balance tasks by total patches
# -----------------------------------------------------------------------------

def chunk_tasks(tasks, n_cpus):
    tasks = sorted(tasks, key=lambda x: x[2], reverse=True)
    loads = [0]*n_cpus
    chunks = [[] for _ in range(n_cpus)]
    for i,j,n in tasks:
        idx = loads.index(min(loads))
        chunks[idx].append((i,j,n))
        loads[idx] += n
    return chunks

# -----------------------------------------------------------------------------
# Build patch->classes JSON for targeted_filter/targeted_matrix modes
# -----------------------------------------------------------------------------

def collect_patch_classes(mask_dir, out_json):
    if os.path.exists(out_json):
        return
    print(f"Collecting classes per mask and storing them at {out_json}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    patch_classes = {}
    for name in os.listdir(mask_dir):
        arr = np.array(Image.open(os.path.join(mask_dir, name)))
        patch_classes[name] = [int(v) for v in np.unique(arr)]
    with open(out_json, 'w') as f:
        json.dump(patch_classes, f)
    print(f"Done with collecting classes per mask")

# -----------------------------------------------------------------------------
# Launch worker pinned to available CPU
# -----------------------------------------------------------------------------

def worker_launcher(script, args_list, cpu_q):
    cpu = cpu_q.get()
    base = os.path.dirname(os.path.abspath(__file__))
    script_fp = os.path.join(base, script)
    cmd = ['taskset', '-c', cpu, sys.executable, script_fp] + args_list
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] {script} on CPU {cpu} failed (code {proc.returncode})")
        print(proc.stdout)
        print(proc.stderr)
        proc.check_returncode()
    cpu_q.put(cpu)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # prepare output dirs
    os.makedirs(cfg['output_dir'], exist_ok=True)
    cpus = cfg['cpus']  # list of ints
    n_cpus = len(cpus)

    # fit normalizer
    norm_pickle = cfg.get('norm_pickle', os.path.join(cfg['output_dir'], 'macenko_normalizer.pkl'))
    fit_normalizer(cfg['m_norm_img'], norm_pickle, cpus)

    # build patch->classes JSON if needed
    if cfg['mode'] in ('targeted_filter','targeted_matrix'):
        collect_patch_classes(
            os.path.join(cfg['data_root'], cfg['mask_dir_name']),
            cfg['patch_classes_json']
        )

    # prepare job chunks
    if cfg['mode'] == 'targeted_matrix':
        combos = read_combos(cfg)
        job_chunks = chunk_tasks(combos, n_cpus)
    else:
        job_chunks = [None]*n_cpus

    # spawn workers
    cpu_q = Queue()
    for c in cpus:
        cpu_q.put(str(c))

    procs = []
    for chunk in job_chunks:
        if not chunk:
            continue
        mode = cfg['mode']
        if mode == 'generalized':
            script = 'worker_generalized.py'
            args_list = [
                '--n_patches', str(cfg['n_patches']),
                '--seed', str(cfg['seed']),
                '--data_root', cfg['data_root'],
                '--mask_dir_name', cfg['mask_dir_name'],
                '--image_dir_name', cfg['image_dir_name'],
                '--norm_pickle', norm_pickle,
                '--stitch_masks', cfg['stitch_masks'],
                '--output_dir', cfg['output_dir']
            ]
        elif mode == 'targeted_filter':
            script = 'worker_targeted_filter.py'
            fp = cfg['targeted_filter_pairs']
            filter_str = ",".join(f"{i}-{j}" for i,j in fp)
            args_list = [
                '--n_patches', str(cfg['n_patches']),
                '--seed', str(cfg['seed']),
                '--filter_pairs', filter_str,
                '--exclude_existing',
                '--patch_classes_json', cfg['patch_classes_json'],
                '--data_root', cfg['data_root'],
                '--mask_dir_name', cfg['mask_dir_name'],
                '--image_dir_name', cfg['image_dir_name'],
                '--norm_pickle', norm_pickle,
                '--stitch_masks', cfg['stitch_masks'],
                '--output_dir', cfg['output_dir']
            ]
        else:  # targeted_matrix
            script = 'worker_targeted_matrix.py'
            args_list = ['--seed', str(cfg['seed']), '--patch_classes_json', cfg['patch_classes_json']]
            for i,j,n in chunk:
                args_list += ['--pair', f"{i},{j}", '--count', str(n)]
            args_list += [
                '--exclude_existing',
                '--data_root', cfg['data_root'],
                '--mask_dir_name', cfg['mask_dir_name'],
                '--image_dir_name', cfg['image_dir_name'],
                '--norm_pickle', norm_pickle,
                '--stitch_masks', cfg['stitch_masks'],
                '--output_dir', cfg['output_dir']
            ]

        p = Process(target=worker_launcher, args=(script, args_list, cpu_q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == '__main__':
    main()