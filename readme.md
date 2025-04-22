# Patch Stitching Repository

A Python repository for generating composite histopathology image patches by stitching two image–mask pairs along  organically shaped binary masks. Supports three modes:

- **Random**: Produce _N_ stitched patches sampled at random. 
- **Filter**: Enrich dataset with underrepresented class co‑occurrences.
- **Matrix**: Generate exactly specified counts for each class–class pairing.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Examples](#examples)
8. [License](#license)

---

## Prerequisites
1. Install python packages needed
- Numpy 
- pickle
- Python Image Library (PIL)
- subprocess and multiprocessing
- pyyaml
- json 
- staintools (see https://github.com/Peter554/StainTools for installation guide)

2. Unpack `BinaryMasks.tar.gz` to a location of your choice to use as stitching masks

---

## Configuration

Each parameter lives in `config.yaml`. Below is a description of every field.

### Shared parameters (all modes)

| Parameter             | Type       | Description                                                                                     |
|-----------------------|------------|-------------------------------------------------------------------------------------------------|
| `mode`                | string     | Operation mode. One of `random`, `filter`, or `matrix`.                                         |
| `seed`                | integer    | Base RNG seed for all sampling, ensuring reproducibility.                                       |
| `cpus`                | list[int]  | CPU core indices for pinning workers (e.g. `[0,1,2]`).                                          |
| `m_norm_img`          | string     | Filepath to a reference image used to fit the Macenko normalizer.                              |
| `data_root`           | string     | Root directory containing your training dataset hierarchy.                                      |
| `mask_dir_name`       | string     | Subfolder under `data_root` where mask PNGs are stored (default: `mask_FINAL`).                 |
| `image_dir_name`      | string     | Subfolder under `data_root` for image JPGs (default: `image`).                                   |
| `stitch_masks`        | string     | Directory of binary stitch masks used for creating boundaries.                                  |
| `output_dir`          | string     | Directory where the generated stitched images (`image/`) and masks (`mask_FINAL/`) will be saved.|
| `norm_pickle`         | string     | _Optional_; path to save/load the fitted Macenko normalizer. Defaults to `<output_dir>/macenko_normalizer.pkl`.|

### Random mode parameters (`mode: random`)

| Parameter     | Type    | Description                                                |
|---------------|---------|------------------------------------------------------------|
| `n_patches`   | integer | Number of random stitched patches to generate.             |

### Filter mode parameters (`mode: filter`)

| Parameter              | Type          | Description                                                            |
|------------------------|---------------|------------------------------------------------------------------------|
| `n_patches`            | integer       | Number of stitched patches to generate.                                 |
| `filter_pairs`         | list[list[int]] | List of 2‑element lists specifying class pairs to include, e.g. `[[1,2],[3,4]]`.|
| `patch_classes_json`   | string        | Path to JSON mapping mask filenames to their contained class IDs.       |
   |

### Matrix mode parameters (`mode: matrix`)

| Parameter              | Type              | Description                                                                                             |
|------------------------|-------------------|---------------------------------------------------------------------------------------------------------|
| `classes`              | list[int]         | _Optional_; list of class IDs matching matrix rows/columns. Defaults to `[0,1,…,len(matrix)-1]`.       |
| `matrix`               | list[list[int]]   | Square 2D list (`NxN`) where cell `[i][j]` is the target number of patches for class pair `(classes[i],classes[j])`. Only `i<j` entries are used. |
| `patch_classes_json`   | string            | Path to JSON mapping mask filenames to their contained class IDs (used to build sampling pools).       |

---

## Usage

### Run Main Launcher

```bash
python main.py --config path/to/config.yaml
```

- The script will:
  1. Fit or load the Macenko normalizer.
  2. Build `patch_classes.json` (for filter/matrix).  
  3. Spawn workers pinned to each CPU.  
  4. Write stitched images to output dir. Masks and images are stored separately


---

## Examples

- Generate 100 random stitched patches:
  ```bash
  python main.py --config configs/random.yaml
  ```

- Generate 100 patches with class‐pairs (1,2) or (3,4):
  ```bash
  python main.py --config configs/filter.yaml
  ```

- Generate exactly the desired matrix combinations:
  ```bash
  python main.py --config configs/matrix.yaml
  ```

## License

Published under the [MIT License](LICENSE).

