import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_analysis(shapes_by_modality, output_dir):
    # Set up the plotting style
    # plt.style.use('seaborn')
    dimensions = ['z', 'y', 'x']
    
    # Create a figure with subplots for each dimension
    fig, axes = plt.subplots(3, 2, figsize=(20, 25))
    fig.suptitle('MRI Image Size Analysis by Modality', fontsize=16, y=1)
    
    # Colors for each modality
    colors = {'adc': 'skyblue', 't2w': 'lightgreen', 'cor': 'salmon', 
              'hbv': 'purple', 'sag': 'orange'}
    
    for dim_idx, dim in enumerate(dimensions):
        # Histogram
        ax_hist = axes[dim_idx, 0]
        for modality in shapes_by_modality:
            sizes = shapes_by_modality[modality][dim]
            ax_hist.hist(sizes, alpha=0.5, label=modality.upper(), 
                        color=colors[modality], bins=20)
        
        ax_hist.set_title(f'{dim.upper()}-Dimension Size Distribution')
        ax_hist.set_xlabel('Size (pixels)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()
        
        # Box plot
        ax_box = axes[dim_idx, 1]
        box_data = [shapes_by_modality[mod][dim] for mod in shapes_by_modality]
        ax_box.boxplot(box_data, labels=[mod.upper() for mod in shapes_by_modality])
        ax_box.set_title(f'{dim.upper()}-Dimension Size Distribution (Box Plot)')
        ax_box.set_ylabel('Size (pixels)')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig.png')
    # Print statistical summary
    print("\nStatistical Summary for Each Modality and Dimension:")
    print("-" * 80)
    
    for modality in shapes_by_modality:
        print(f"\n{modality.upper()} Modality:")
        for dim in dimensions:
            sizes = shapes_by_modality[modality][dim]
            stats = {
                'count': len(sizes),
                'min': min(sizes),
                'max': max(sizes),
                'mean': np.mean(sizes),
                'median': np.median(sizes),
                'std': np.std(sizes)
            }
            print(f"\n{dim.upper()}-Dimension:")
            print(f"Count: {stats['count']}")
            print(f"Range: {stats['min']} - {stats['max']}")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Median: {stats['median']:.2f}")
            print(f"Standard Deviation: {stats['std']:.2f}")

def combine_and_plot_results(output_dir):
    # Initialize combined results
    combined_results = None
    
    # Load and combine all pickle files
    output_dir = Path(output_dir)
    for pkl_file in output_dir.glob('shapes_chunk_*.pkl'):
        with open(pkl_file, 'rb') as f:
            chunk_results = pickle.load(f)
            if combined_results is None:
                combined_results = chunk_results
            else:
                for modality in chunk_results:
                    for dim in chunk_results[modality]:
                        combined_results[modality][dim].extend(chunk_results[modality][dim])
    
    # Plot the combined results (using the plotting code from previous example)
    plot_analysis(combined_results, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge picai_shape_analysis.py chunks and plot shape distributions")
    parser.add_argument("--output_dir", default="outputs", help="Directory holding shapes_chunk_*.pkl")
    combine_and_plot_results(parser.parse_args().output_dir)