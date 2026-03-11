import argparse
import os
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
import gc
from torchvision.utils import save_image


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from discrete_mbrl.data_helpers import prepare_dataloaders
from discrete_mbrl.model_construction import construct_ae_model
from discrete_mbrl.env_helpers import make_env


def analyze_environment_complexity(env_name):
    """Analyze visual complexity of different environments"""

    print(f"\n=== ENVIRONMENT ANALYSIS: {env_name} ===")

    # Sample some episodes to see variety
    env = make_env(env_name)
    unique_observations = []

    for episode in range(10):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
        unique_observations.append(obs)

        for step in range(20):
            action = env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            unique_observations.append(obs)
            if done:
                break

    # Analyze color diversity
    all_pixels = np.concatenate([obs.flatten() for obs in unique_observations])
    unique_colors = len(np.unique(all_pixels))

    print(f"Unique pixel values: {unique_colors}")
    print(f"Observations collected: {len(unique_observations)}")
    print(f"Observation shape: {unique_observations[0].shape}")

    return unique_observations


def analyze_codebook_usage(encodings_list):
    """Enhanced analysis of codebook usage patterns"""

    print("\n=== CODEBOOK USAGE ANALYSIS ===")

    # Flatten all encodings
    all_codes = []
    for encodings in encodings_list:
        if len(encodings.shape) > 1:
            all_codes.extend(encodings.flatten().tolist())
        else:
            all_codes.extend(encodings.tolist())

    # Count usage
    from collections import Counter
    code_counts = Counter(all_codes)
    total_positions = len(all_codes)

    # Basic stats
    unique_codes = len(code_counts)
    max_possible_codes = max(all_codes) + 1 if all_codes else 0
    utilization = (unique_codes / max_possible_codes * 100) if max_possible_codes > 0 else 0

    print(f"Total codes used: {unique_codes} out of {max_possible_codes}")
    print(f"Utilization: {utilization:.1f}%")
    print(f"Code range: {min(all_codes)} to {max(all_codes)}")

    # Most frequent codes
    print(f"Most frequent codes (code: count, percentage):")
    for code, count in code_counts.most_common(10):
        percentage = (count / total_positions) * 100
        print(f"  Code {code}: {count} positions ({percentage:.1f}%)")

    return code_counts


def analyze_spatial_distribution(encodings_list, num_samples=3):
    """Analyze how codes are distributed spatially"""

    print(f"\n=== SPATIAL CODE DISTRIBUTION ===")

    for i, encodings in enumerate(encodings_list[:num_samples]):
        print(f"Sample {i + 1} code map:")
        print(f"  Shape: {encodings.shape}")

        if len(encodings.shape) > 1:
            flat_codes = encodings.flatten()
        else:
            flat_codes = encodings

        from collections import Counter
        spatial_counts = Counter(flat_codes.tolist())
        total_positions = len(flat_codes)

        for code, count in spatial_counts.most_common(10):
            percentage = (count / total_positions) * 100
            print(f"    Code {code}: {count}/{total_positions} positions ({percentage:.1f}%)")


def analyze_code_meanings(semantic_map):
    """Analyze what each code represents based on visual properties"""

    print(f"\n=== CODE SEMANTIC ANALYSIS ===")

    code_analysis = {}

    for code, data in semantic_map.items():
        if not data['samples']:
            continue

        # Calculate visual properties
        samples = torch.stack(data['samples'])
        originals = torch.stack(data['originals']) if data['originals'] else samples

        # Average brightness (normalized to 0-1)
        avg_brightness = samples.mean().item()
        orig_brightness = originals.mean().item()

        # Variety (standard deviation)
        variety = samples.std().item()
        orig_variety = originals.std().item()

        # Color distribution (if RGB)
        if samples.shape[1] >= 3:
            red_avg = samples[:, 0].mean().item()
            green_avg = samples[:, 1].mean().item()
            blue_avg = samples[:, 2].mean().item()
            dominant_color = max([('Red', red_avg), ('Green', green_avg), ('Blue', blue_avg)],
                                 key=lambda x: x[1])
        else:
            dominant_color = ('Grayscale', avg_brightness)

        # Classify code type
        code_type = "UNKNOWN"
        if avg_brightness < 0.1:
            code_type = "EMPTY SPACE"
        elif avg_brightness > 0.7 and variety < 0.1:
            code_type = "UNIFORM BRIGHT"
        elif variety > 0.2:
            code_type = "COMPLEX CONTENT"
        elif 0.2 < avg_brightness < 0.6:
            code_type = "MEDIUM OBJECTS"
        else:
            code_type = "UNIFORM OBJECTS"

        analysis = {
            'brightness': avg_brightness,
            'variety': variety,
            'orig_brightness': orig_brightness,
            'orig_variety': orig_variety,
            'samples': len(data['samples']),
            'dominant_color': dominant_color,
            'type': code_type
        }

        code_analysis[code] = analysis

        print(f"Code {code}: {code_type}")
        print(f"  Samples: {len(data['samples'])}")
        print(f"  Brightness: {avg_brightness:.3f} (orig: {orig_brightness:.3f})")
        print(f"  Variety: {variety:.3f} (orig: {orig_variety:.3f})")
        print(f"  Dominant: {dominant_color[0]} ({dominant_color[1]:.3f})")
        print()

    return code_analysis


def create_semantic_map_optimized(encoder, train_loader, device, max_samples_per_code=50):
    """
    Optimized function to create a semantic map using efficient tensor operations.
    """
    num_codes = encoder.quantizer._num_embeddings
    semantic_map = {i: {'samples': [], 'originals': []} for i in range(num_codes)}
    needed_codes = set(range(num_codes))

    print(f"Processing data with max {max_samples_per_code} samples per code...")
    print(f"Codebook size: {num_codes}")

    # Track all encodings for analysis
    all_encodings = []

    with torch.no_grad():
        for batch_idx, (batch, _, _, _, _) in enumerate(train_loader):
            if not needed_codes:
                print("All necessary samples collected. Stopping early.")
                break

            batch = batch.to(device)
            try:
                recon_batch, _, _, encodings = encoder(batch)

                print(f"Encodings shape: {encodings.shape}")
                print(f"Encodings dtype: {encodings.dtype}")
                print(f"Encodings min/max: {encodings.min().item():.3f}/{encodings.max().item():.3f}")

                # Convert one-hot to indices if needed
                if encodings.dtype == torch.float32 and len(encodings.shape) == 3:
                    # One-hot format: (batch, num_codes, spatial_positions)
                    encodings_indices = encodings.argmax(dim=1)  # (batch, spatial_positions)
                    print(f"Converted one-hot to indices, shape: {encodings_indices.shape}")
                else:
                    # Already indices
                    encodings_indices = encodings

                # Store for analysis
                all_encodings.append(encodings_indices.cpu())

                encodings_flat = encodings_indices.reshape(-1)
                unique_codes_in_batch = torch.unique(encodings_flat).cpu().numpy()

                print(f"Unique codes in first batch: {len(unique_codes_in_batch)}")
                print(f"Code range: {unique_codes_in_batch.min()} to {unique_codes_in_batch.max()}")

                codes_to_process = needed_codes.intersection(unique_codes_in_batch)

                for code in codes_to_process:
                    # Find where this code appears in the flattened encodings
                    indices = (encodings_flat == code).nonzero(as_tuple=True)[0]

                    # Determine how many samples we still need for this code
                    needed_now = max_samples_per_code - len(semantic_map[code]['samples'])

                    # Take the minimum of what's needed and what's available
                    indices_to_take = indices[:needed_now]

                    # Map flat indices back to batch indices
                    spatial_size = encodings_indices.shape[1]
                    batch_indices = (indices_to_take // spatial_size).long()

                    if batch_indices.numel() > 0:
                        unique_batch_indices = torch.unique(batch_indices)
                        semantic_map[code]['samples'].extend(list(recon_batch[unique_batch_indices].cpu()))
                        semantic_map[code]['originals'].extend(list(batch[unique_batch_indices].cpu()))

                    # If we have enough for this code, we don't need to look for it anymore
                    if len(semantic_map[code]['samples']) >= max_samples_per_code:
                        needed_codes.discard(code)

                if batch_idx % 5 == 0:
                    active_codes = len([k for k, v in semantic_map.items() if v['samples']])
                    total_samples = sum(len(v['samples']) for v in semantic_map.values())
                    print(
                        f"Batch {batch_idx}: Found {active_codes} unique codes so far, collected {total_samples} total samples.")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    # Analyze codebook usage
    analyze_codebook_usage(all_encodings)
    analyze_spatial_distribution(all_encodings)

    final_map = {k: v for k, v in semantic_map.items() if v['samples']}

    print(f"Total unique codes discovered: {len(set().union(*[enc.flatten().tolist() for enc in all_encodings]))}")
    print(f"Final semantic map: {len(final_map)} unique codes")

    return final_map


def visualize_prototypes(semantic_map, output_dir):
    """Visualizes the 'prototype' (average) observation for each discrete code."""
    print("  - Visualizing prototypes...")
    proto_dir = os.path.join(output_dir, 'prototypes')
    os.makedirs(proto_dir, exist_ok=True)
    for code, data in semantic_map.items():
        obs_list = data['samples']
        if not obs_list:
            continue
        try:
            avg_obs = torch.stack(obs_list).mean(dim=0).numpy()
            avg_obs = np.transpose(avg_obs, (1, 2, 0))
            avg_obs = np.clip(avg_obs, 0, 1)

            # Handle grayscale
            if avg_obs.shape[2] == 1:
                avg_obs = avg_obs.squeeze(2)

            plt.figure(figsize=(4, 4))
            plt.imshow(avg_obs, cmap='gray' if len(avg_obs.shape) == 2 else None)
            plt.title(f'Code {code} Prototype')
            plt.axis('off')
            plt.savefig(os.path.join(proto_dir, f'code_{code}.png'), bbox_inches='tight', dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error visualizing prototype for code {code}: {e}")


def visualize_samples(semantic_map, output_dir, num_samples=16):
    """Visualizes a grid of random sample observations for each discrete code."""
    print("  - Visualizing samples...")
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    for code, data in semantic_map.items():
        obs_list = data['samples']
        if not obs_list:
            continue
        try:
            count = min(len(obs_list), num_samples)
            grid = torch.stack(obs_list[:count])
            save_image(grid, os.path.join(samples_dir, f'code_{code}.png'),
                       normalize=True, nrow=int(np.sqrt(count)))
        except Exception as e:
            print(f"Error visualizing samples for code {code}: {e}")


def visualize_originals(semantic_map, output_dir, num_samples=16):
    """Visualizes a grid of original input images for each discrete code."""
    print("  - Visualizing original inputs...")
    originals_dir = os.path.join(output_dir, 'originals')
    os.makedirs(originals_dir, exist_ok=True)
    for code, data in semantic_map.items():
        obs_list = data['originals']
        if not obs_list:
            continue
        try:
            count = min(len(obs_list), num_samples)
            grid = torch.stack(obs_list[:count])
            save_image(grid, os.path.join(originals_dir, f'code_{code}.png'),
                       normalize=True, nrow=int(np.sqrt(count)))
        except Exception as e:
            print(f"Error visualizing originals for code {code}: {e}")


def create_summary_report(semantic_map, code_analysis, output_dir):
    """Create a summary report of the analysis"""

    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=== VQ-VAE CODEBOOK ANALYSIS REPORT ===\n\n")

        # Overall stats
        total_codes = len(semantic_map)
        f.write(f"Total active codes: {total_codes}\n")
        f.write(f"Total samples collected: {sum(len(v['samples']) for v in semantic_map.values())}\n\n")

        # Code breakdown by type
        type_counts = {}
        for analysis in code_analysis.values():
            code_type = analysis['type']
            type_counts[code_type] = type_counts.get(code_type, 0) + 1

        f.write("Code types:\n")
        for code_type, count in sorted(type_counts.items()):
            f.write(f"  {code_type}: {count} codes\n")
        f.write("\n")

        # Detailed breakdown
        f.write("Detailed code analysis:\n")
        for code in sorted(semantic_map.keys()):
            if code in code_analysis:
                analysis = code_analysis[code]
                f.write(f"Code {code}: {analysis['type']}\n")
                f.write(f"  Samples: {analysis['samples']}\n")
                f.write(f"  Brightness: {analysis['brightness']:.3f}\n")
                f.write(f"  Variety: {analysis['variety']:.3f}\n")
                f.write(f"  Dominant: {analysis['dominant_color'][0]}\n\n")

    print(f"Analysis report saved to {report_path}")


def main(args):
    """Main function to run the semantic visualization experiment."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze environment complexity
    complexity_analysis = analyze_environment_complexity(args.env_name)

    print("Loading data...")
    train_loader, _, _ = prepare_dataloaders(
        env_name=args.env_name, batch_size=args.batch_size, n=args.max_transitions,
        preprocess=True, preload_all=False, n_preload=0
    )
    print("Data loaded successfully.")

    # Check reverse transform
    if hasattr(train_loader.dataset, 'flat_rev_obs_transform'):
        print(f"Reverse transform function: {train_loader.dataset.flat_rev_obs_transform}")

    print("Determining input shape from data...")
    first_obs_batch = next(iter(train_loader))[0]
    input_dim = first_obs_batch.shape[1:]
    print(f"Detected input shape: {input_dim}")
    del first_obs_batch
    gc.collect()

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = Dotdict({
        "env_name": args.env_name, "device": device, "ae_model_type": "vqvae",
        "embedding_dim": args.embedding_dim, "filter_size": args.filter_size,
        "ae_model_version": args.ae_model_version, "codebook_size": args.codebook_size,
        "latent_dim": args.latent_dim, "ae_grad_clip": 1.0, "learning_rate": 0.0003, "wandb": False
    })

    try:
        encoder, _ = construct_ae_model(input_dim=input_dim, args=model_args, load=False)
        encoder.load_state_dict(torch.load(args.model_path, map_location=device))
        encoder.to(device)
        encoder.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Creating semantic map...")
    semantic_map = create_semantic_map_optimized(encoder, train_loader, device, max_samples_per_code=50)

    # Analyze code meanings
    code_analysis = analyze_code_meanings(semantic_map)

    print("Semantic map created.")

    print("Generating visualizations...")
    visualize_prototypes(semantic_map, args.output_dir)
    visualize_samples(semantic_map, args.output_dir)
    visualize_originals(semantic_map, args.output_dir)

    # Create summary report
    create_summary_report(semantic_map, code_analysis, args.output_dir)

    print(f"Visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VQ-VAE model.')
    parser.add_argument('--env_name', type=str, default='minigrid-crossing-stochastic', help='Name of the environment.')
    parser.add_argument('--output_dir', type=str, default='semantic_visuals', help='Directory to save visualizations.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing data.')
    parser.add_argument('--max_transitions', type=int, default=10000, help='Max transitions to load.')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--ae_model_version', type=str, default='2')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=81)
    args = parser.parse_args()
    main(args)