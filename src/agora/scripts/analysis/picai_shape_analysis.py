# save as analyze_shapes.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
import os

import torchio as tio
import torch
import SimpleITK as sitk
target_spacing=(1.0, 1.0, 1.0)
target_shape=[26, 96,112]

transforms = tio.Compose([
# tio.Resample(target_spacing),
tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)),
# tio.CropOrPad(target_shape=target_shape, padding_mode='minimum')
])       


def read_and_preprocess( image_path, roi_path,preprocess:bool) -> np.ndarray:
    """Read and preprocess a single image."""
    if isinstance(image_path, Path):
        image_path = image_path.as_posix()
    
    if not preprocess:
         pass
    else:
        if isinstance(roi_path, Path):
            roi_path = roi_path.as_posix()
        
    # Read image

    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)


    
    # Apply transforms
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(image).unsqueeze(0))
    )
    # print(subject.shape)
    transformed = transforms(subject)
    # print(transformed.shape)
    return transformed.image.data.squeeze().numpy()

def process_chunk(chunk_df, root_dir, output_dir):
    modalities = ['adc', 't2w', 'cor', 'hbv', 'sag']
    shapes_by_modality = {mod: {'z': [], 'y': [], 'x': []} for mod in modalities}
    
    for idx, row in chunk_df.iterrows():
        for modality in modalities:
            image_path = root_dir / f"{row['patient_id']}/{row['patient_id']}_{row['study_id']}_{modality}.mha"
            if image_path.exists():
                image = read_and_preprocess(image_path=image_path, roi_path="", preprocess=False)
                shapes_by_modality[modality]['z'].append(image.shape[0])
                shapes_by_modality[modality]['y'].append(image.shape[1])
                shapes_by_modality[modality]['x'].append(image.shape[2])
    
    # Save results
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    output_file = output_dir / f"shapes_chunk_{task_id}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(shapes_by_modality, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--total_tasks', type=int, required=True)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    csv_data = pd.read_csv(args.csv_path)
    
    # Get task ID from SLURM
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    
    # Split data into chunks
    chunk_size = len(csv_data) // args.total_tasks + 1
    start_idx = task_id * chunk_size
    end_idx = min((task_id + 1) * chunk_size, len(csv_data))
    chunk_df = csv_data.iloc[start_idx:end_idx]
    
    # Process chunk
    process_chunk(chunk_df, Path(args.root_dir), output_dir)

if __name__ == "__main__":
    main()