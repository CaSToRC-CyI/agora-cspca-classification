import SimpleITK as sitk
import numpy as np
import os
import pickle
import argparse
from pathlib import Path


def get_roi(roi_path, padding=(0, 0, 0)):
    roi = sitk.ReadImage(roi_path)
    roi_array = sitk.GetArrayFromImage(roi)
    binary_roi_array = (roi_array > 0).astype(np.uint8)
    # binary_roi = sitk.GetImageFromArray(binary_roi_array)
    # binary roi is not needed for sitk.ConnectedComponent since
    # non-zero pixels are considered to be objects, zero-valued pixels are considered to be background
    # This includes the 1 and 2 for prostate regions
    image_shape = roi_array.shape

    # Get image dimensions
    depth, height, width = image_shape

    # 1. Convert binary image into a connected component image, each component has an integer label.
    # 2. Relabel components so that they are sorted according to size (there is an
    #    optional minimumObjectSize parameter to get rid of small components).
    # 3. Get largest connected componet, label==1 in sorted component image.

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    component_image = cc_filter.Execute(roi)
    sorted_component_image = sitk.RelabelComponent(
        component_image, sortByObjectSize=True
    )
    largest_component = sorted_component_image == 1

    # Convert back to numpy
    clean_mask = sitk.GetArrayFromImage(largest_component).astype(np.uint8)

    # Find ROI bounds
    z_indices, y_indices, x_indices = np.where(clean_mask == 1)
    if len(z_indices) == 0:
        return roi_array  # Return original if no ROI found

    # Calculate bounds with padding
    z_min = max(0, z_indices.min() - padding[0])
    z_max = min(depth, z_indices.max() + 1 + padding[0])
    y_min = max(0, y_indices.min() - padding[1])
    y_max = min(height, y_indices.max() + 1 + padding[1])
    x_min = max(0, x_indices.min() - padding[2])
    x_max = min(width, x_indices.max() + 1 + padding[2])

    return z_min, z_max, y_min, y_max, x_min, x_max


def process_chunk(file_list, output_dir, task_id):
    shapes_by_dimension = {"z": [], "y": [], "x": []}

    for roi_path in file_list:
        try:
            # Padding was previously (3, 50, 50)
            z_min, z_max, y_min, y_max, x_min, x_max = get_roi(roi_path)
            shapes_by_dimension["z"].append(z_max - z_min)
            shapes_by_dimension["y"].append(y_max - y_min)
            shapes_by_dimension["x"].append(x_max - x_min)
        except Exception as e:
            print(f"Error processing {roi_path}: {e}")

    # Save results
    output_file = output_dir / f"goc_roi_shapes_chunk_{task_id}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(shapes_by_dimension, f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyze ROI shapes for any resampled dataset"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_tasks", type=int, required=True)
    return parser.parse_args()


def main(input_dir, output_dir, total_tasks):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    roi_files = list(input_dir.glob("*.nii.gz"))

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    chunk_size = len(roi_files) // total_tasks + 1
    start_idx = task_id * chunk_size
    end_idx = min((task_id + 1) * chunk_size, len(roi_files))
    file_chunk = roi_files[start_idx:end_idx]

    process_chunk(file_chunk, output_dir, task_id)
    print("Finished analysis!")


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
