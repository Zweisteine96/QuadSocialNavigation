import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.patches as patches
from collections import defaultdict
import pandas as pd
import sys

# Assume your model definition is in this file
from social_vae import SocialVAE 

# --- CONFIGURATIONS --- #
OB_HORIZON = 8          # Number of past frames for history
PRED_HORIZON = 12       # Number of future frames to predict
DELTA_T = 0.4           # Time step for ETH/UCY datasets (1 / 2.5 Hz)
BOX_SIZE = 0.3          # Visual size of the agent's box
TEXT_OFFSET = 0.2       # Visual offset for the ID text
MISSING_THRESHOLD = 3   # Frames to wait before removing a disappeared agent
OB_RADIUS = 2
DELTA_T = 0.4
RNN_HIDDEN_DIM = 256

# --- UTILITY FUNCTIONS --- #
def load_model(ckpt_path, device):
    """Loads the trained SocialVAE model from a checkpoint."""
    print(f"--> Loading model from checkpoint '{ckpt_path}'...")
    try:
        model = SocialVAE(
            horizon=PRED_HORIZON,
            ob_radius=OB_RADIUS,
            hidden_dim=RNN_HIDDEN_DIM
        )
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model'])
        model.to(torch.device('cpu'))
        model.eval()
        print("--> Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_and_process_data_with_derivatives(file_path):
    """
    Loads raw trajectory data and processes it into a single tensor with
    positions, velocities, and accelerations.
    """
    print(f"--> Loading and processing data from '{file_path}'...")
    try:
        df = pd.read_csv(
            file_path, sep='\s+', header=None,
            names=['frame_id', 'person_id', 'x', 'y'],
            dtype={'frame_id': int}
        )
        
        # Get unique IDs and create mappings for array indexing
        unique_person_ids = sorted(df['person_id'].unique())
        unique_frame_ids = sorted(df['frame_id'].unique())
        
        person_id_map = {pid: i for i, pid in enumerate(unique_person_ids)}
        frame_id_map = {fid: i for i, fid in enumerate(unique_frame_ids)}
        
        num_persons = len(unique_person_ids)
        num_frames = len(unique_frame_ids)
        
        # Initialize arrays for positions and final features
        positions = np.full((num_persons, num_frames, 2), np.nan, dtype=float)
        features = np.full((num_persons, num_frames, 6), np.nan, dtype=float)
        
        # Populate the position array first
        for _, row in df.iterrows():
            p_idx = person_id_map[row['person_id']]
            f_idx = frame_id_map[row['frame_id']]
            positions[p_idx, f_idx] = [row['x'], row['y']]
            
        features[:, :, 0:2] = positions

        # Calculate derivatives for each person
        # NOTE: socialVAE needs vel and acc as inputs as well
        print("--> Calculating velocities and accelerations...")
        for p_idx in range(num_persons):
            # Find contiguous segments of data (where there are no NaNs)
            person_pos = positions[p_idx]
            # Create a mask to identify where valid data exists
            mask = ~np.isnan(person_pos).any(axis=1)
            # segments = np.where(np.diff(mask) != 0)[0] + 1
            # segment_indices = np.split(
            #     np.arange(len(mask))[mask], 
            #     np.searchsorted(np.arange(len(mask))[mask], segments)
            # )
            indices = np.where(mask)[0]
            segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            
            #for seg_idxs in segment_indices:
            # Process each continuous trajectory segment separately
            for seg_idxs in segments:
                if len(seg_idxs) < 2: continue # Need at least 2 points for derivatives
                
                # Calculate v and a for this continuous segment
                seg_pos = person_pos[seg_idxs]
                # NOTE: different from what socialVAE did in their data processing pipeline!
                seg_v = np.gradient(seg_pos, DELTA_T, axis=0)
                seg_a = np.gradient(seg_v, DELTA_T, axis=0)
                
                # Place calculated features back into the main array
                features[p_idx, seg_idxs, 2:4] = seg_v
                features[p_idx, seg_idxs, 4:6] = seg_a

        print(f"--> Data processing complete.")
        return features, person_id_map, frame_id_map, unique_frame_ids

    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


def run_animation_with_predictions(
        model, features, person_map, frame_map, frame_ids, device, args
    ):
    """Main animation function with stateful prediction plotting."""
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    ax.axis('equal') # x and y axes have the same scale

    #valid_coords = features[~np.isnan(features).any(axis=-1)]
    valid_coords = features[~np.all(np.isnan(features), axis=(1, 2))][:, :, 0:2]
    valid_coords = valid_coords[~np.isnan(valid_coords).any(axis=-1)]
    ax.set_xlim(valid_coords[:, 0].min() - 2, valid_coords[:, 0].max() + 2)
    ax.set_ylim(valid_coords[:, 1].min() - 2, valid_coords[:, 1].max() + 2)
    
    # State management for all visual artists
    person_artists = {}
    colors = plt.colormaps.get_cmap('tab20')
    
    # Create dummy lines for the legend
    ax.plot([], [], color='black', marker='o', label='History Trail')
    ax.plot([], [], color='red', linestyle='--', marker='^', label='Prediction')
    ax.legend(loc='upper right')
    title = ax.set_title("Initializing...")
    
    person_idx_to_id = {v: k for k, v in person_map.items()}

    # Animation update function
    def update(frame_index):
        """This function is called for each frame of the animation."""
        current_frame_id = frame_ids[frame_index]
        title.set_text(f"Live Prediction at Frame ID: {current_frame_id}")

        # --- 1. Identify agents for prediction and their data ---
        agents_for_prediction = [] # list of persons who have a valid history
        history_windows = [] # list of corresponding history data
        pids_in_frame = set() # set of all persons present in this frame

        # Iterate through all people to see who is active and who can be predicted
        for p_idx in range(features.shape[0]):
            # Check if the current person has a valid position right now
            current_pos = features[p_idx, frame_index, 0:2]
            if not np.isnan(current_pos).any():
                pids_in_frame.add(person_idx_to_id[p_idx])
            
            # Check for a complete history window up to the current frame
            if frame_index >= OB_HORIZON - 1:
                # Slice the history window from the main feature array
                hist_window = features[p_idx, frame_index - OB_HORIZON + 1 : frame_index + 1, :]
                # Check if the window has missing data, if no, it's valid for prediction
                if not np.isnan(hist_window).any():
                    agents_for_prediction.append(person_idx_to_id[p_idx])
                    history_windows.append(hist_window)

        # --- 2. Batched Inference ---
        predictions = {}
        if agents_for_prediction:
            hist_batch = np.stack(history_windows)
            x = torch.from_numpy(hist_batch).permute(1, 0, 2).float().to(device) # second dim is actually the number of persons
            print(f"x shape: {x.shape}")
            
            # For visualization, we can use a simplified neighbor tensor
            # In a real test setting, neighbors would be computed properly.
            batch_size = x.shape[1]
            #dummy_neighbors = torch.full((OB_HORIZON, batch_size, 1, 6), 1e9, device=device)
            neighbor_list = [torch.cat([x[:, :i, :], x[:, i+1:, :]], dim=1) for i in range(batch_size)]
            neighbors = torch.stack(neighbor_list).permute(1, 0, 2, 3)
            print(f"neighbor shape: {neighbors.shape}")

            with torch.no_grad():
                # At this frame, we predict for each person who has a valid history window
                # NOTE: seems not really being used for inference? Can even replace neighbors with None
                y_pred_samples = model(x, neighbors, n_predictions=1)
                print(f"y_pred_samples shape: {y_pred_samples.shape}")
                y_pred_batch = y_pred_samples[0].cpu().numpy()
            
            # Map the batched prediction results back to their respective persons
            for i, pid in enumerate(agents_for_prediction):
                predictions[pid] = y_pred_batch[:, i, :]

        # --- 3. Artist Management ---
        updated_artists = [title]
        
        # Loop through all agents ever seen to update, hide, or create them
        all_pids_to_manage = pids_in_frame.union(person_artists.keys())
        
        for pid in all_pids_to_manage:
            p_idx = person_map[pid]
            current_pos = features[p_idx, frame_index, 0:2]

            if pid not in person_artists:
                # Found a NEW person in this frame. Create all their visual artists.
                color = colors(pid % 20)
                box = patches.Rectangle((0,0), BOX_SIZE, BOX_SIZE, facecolor=color, edgecolor='black', zorder=10)
                text = ax.text(0, 0, str(pid), color='black', fontsize=9, fontweight='bold', va='center', zorder=11)
                history_line, = ax.plot([], [], color=color, linewidth=2, marker='o', markersize=2)
                pred_line, = ax.plot([], [], color='red', linestyle='--', marker='^', markersize=4, alpha=0.4)
                ax.add_patch(box)
                person_artists[pid] = {
                    'box': box, 'text': text, 'history_line': history_line, 
                    'pred_line': pred_line, 'positions': [], 'missing_frames': 0
                }

            artists = person_artists[pid]
            
            if pid in pids_in_frame:
                # This agent is active in this frame
                artists['missing_frames'] = 0 # reset disappearance counter
                artists['positions'].append(current_pos) # add current position to the history trail
                if len(artists['positions']) > 20: # keep trail length manageable
                    artists['positions'].pop(0)

                # Update visual elements
                artists['box'].set_xy((current_pos[0] - BOX_SIZE/2, current_pos[1] - BOX_SIZE/2))
                artists['text'].set_position((current_pos[0] + TEXT_OFFSET, current_pos[1]))
                pos_array = np.array(artists['positions'])
                artists['history_line'].set_data(pos_array[:, 0], pos_array[:, 1])
                
                # Check if a prediction was made for this person
                if pid in predictions:
                    pred_traj = predictions[pid]
                    # Update the prediction line data, starting from the current position
                    artists['pred_line'].set_data(
                        np.insert(pred_traj[:, 0], 0, current_pos[0]),
                        np.insert(pred_traj[:, 1], 0, current_pos[1])
                    )
                    artists['pred_line'].set_visible(True)
                else:
                    # If no prediction was made (e.g., insufficient history), hide the line
                    artists['pred_line'].set_visible(False)
                
                # for artist in artists.values():
                #     if hasattr(artist, 'set_visible'): artist.set_visible(True)
                # Make sure all artists for this person are visible.
                for key, artist in artists.items():
                    if hasattr(artist, 'set_visible'): artist.set_visible(True)
                
                #updated_artists.extend(list(artists.values())[:-2]) # Don't add positions/missing_frames
                # Add all visual artists to the list to be returned for blitting.
                updated_artists.extend([v for k, v in artists.items() if k not in ['positions', 'missing_frames']])

            else:
                # Agent has disappeared
                artists['missing_frames'] += 1
                # Hide all their visual elements.
                for key, artist in artists.items():
                    if hasattr(artist, 'set_visible'): artist.set_visible(False)
                # for artist in artists.values():
                #     if hasattr(artist, 'set_visible'): artist.set_visible(False)
        
        # Cleanup loop
        # pids_to_remove = [pid for pid, artists in person_artists.items() if artists['missing_frames'] > MISSING_THRESHOLD]
        # for pid in pids_to_remove:
        #     for artist in person_artists[pid].values():
        #         if hasattr(artist, 'remove'): artist.remove()
        #     del person_artists[pid]
        # --- Cleanup Loop ---
        # Find all agents who have been missing for too long.
        pids_to_remove = [pid for pid, artists in person_artists.items() if artists['missing_frames'] > MISSING_THRESHOLD]
        for pid in pids_to_remove:
            # Permanently remove their artists from the plot.
            artist_keys_to_remove = ['box', 'text', 'history_line', 'pred_line']
            for key in artist_keys_to_remove:
                person_artists[pid][key].remove()
            # Remove the agent from the state dictionary.
            del person_artists[pid]
            
        return updated_artists

    print("--> Starting live animation...")
    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000/args.fps, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Live visualization of SocialVAE predictions with stateful agent animation."
    )
    parser.add_argument('--model_ckpt', required=True, type=str, help="Path to the trained model checkpoint.")
    parser.add_argument('--data_file', default='biwi_eth.txt', type=str, help="Path to the raw ETH/UCY format dataset file.")
    parser.add_argument('--device', default='cpu', type=str, help="Device to run the model on ('cpu' or 'cuda').")
    parser.add_argument('--fps', type=int, default=5, help="Frames per second for the animation. Lower is slower.")
    args = parser.parse_args()

    #device = torch.device(args.device)
    device  =torch.device('cpu')
    model = load_model(args.model_ckpt, device)
    features, person_map, frame_map, frame_ids = load_and_process_data_with_derivatives(args.data_file)
    run_animation_with_predictions(
        model, features, person_map, frame_map, frame_ids, device, args
    )