import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import sys
import argparse  # Import the argparse library

# --- Configuration ---
FILE_PATH = 'data/eth/test/biwi_eth.txt'
OUTPUT_FILENAME = 'eth_trajectory_animation.mp4'
BOX_SIZE = 0.3  # The size of the box representing each person
TEXT_OFFSET = 0.1
MISSING_THRESHOLD = 3  # Number of frames a person can be missing before being removed
FPS = 5 # Frames per second for the output video

def load_and_group_data(file_path):
    """Loads the dataset and groups it by frame ID for easy iteration."""
    try:
        column_names = ['frame_id', 'person_id', 'x', 'y']
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,
            header=None,
            names=column_names,
            dtype={'frame_id': int} # Ensure correct types
        )
        # Group data by frame_id and store in a dictionary for fast lookups
        grouped_data = {frame: group for frame, group in df.groupby('frame_id')}
        # Get a sorted list of all frame IDs to define the animation sequence
        sorted_frame_ids = sorted(grouped_data.keys())
        return grouped_data, sorted_frame_ids
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

def create_trajectory_animation(show_live=False):
    """
    Main function to create the trajectory animation.

    Args:
        show_live (bool): If True, shows the animation live. Otherwise, saves to a file.
    """
    
    # 1. Load and prepare data
    grouped_by_frame, frame_ids = load_and_group_data(FILE_PATH)
    
    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    ax.axis('equal') # Ensure aspect ratio is correct

    # Determine global plot limits by checking all coordinates
    all_x = np.concatenate([g['x'].values for g in grouped_by_frame.values()])
    all_y = np.concatenate([g['y'].values for g in grouped_by_frame.values()])
    ax.set_xlim(all_x.min() - 2, all_x.max() + 2)
    ax.set_ylim(all_y.min() - 2, all_y.max() + 2)

    # 3. State management dictionary
    # This will store the state of every person currently on screen.
    person_data = {}

    # Create a color map to assign a unique color to each person
    num_unique_persons = pd.read_csv(FILE_PATH, delim_whitespace=True, header=None)[1].nunique()
    colors = cm.get_cmap('tab20', num_unique_persons)

    def get_color_for_person(person_id):
        # Assign a consistent color based on the person's ID
        return colors(person_id % num_unique_persons)

    def update(frame_id):
        """This function is called by FuncAnimation for each frame."""
        ax.set_title(f"Trajectory from {FILE_PATH}, Frame ID: {frame_id}")
        
        current_frame_df = grouped_by_frame.get(frame_id, pd.DataFrame())
        pids_in_current_frame = set(current_frame_df['person_id'])
        
        # A. Update existing people and add new ones
        for _, row in current_frame_df.iterrows():
            pid, x, y = int(row['person_id']), row['x'], row['y']
            
            if pid not in person_data:
                # NEW person found
                color = get_color_for_person(pid)
                box = patches.Rectangle(
                    (x - BOX_SIZE / 2, y - BOX_SIZE / 2),
                    BOX_SIZE, BOX_SIZE,
                    facecolor=color, edgecolor='black'
                )
                ax.add_patch(box)
                line, = ax.plot([], [], color=color, linewidth=2)

                text = ax.text(
                    x + TEXT_OFFSET, y, str(pid), 
                    color='black', fontsize=8, fontweight='bold', va='center'
                )
                
                person_data[pid] = {
                    'box': box, 'line': line, 'text': text,
                    'positions': [],
                    'missing_frames': 0, 'color': color
                }
            
            # Update data for both new and existing people
            person_data[pid]['missing_frames'] = 0
            person_data[pid]['positions'].append((x, y))
            person_data[pid]['box'].set_xy((x - BOX_SIZE / 2, y - BOX_SIZE / 2))
            person_data[pid]['box'].set_visible(True)
            person_data[pid]['text'].set_position((x + TEXT_OFFSET, y))
            person_data[pid]['text'].set_visible(True)
            
            positions_array = np.array(person_data[pid]['positions'])
            person_data[pid]['line'].set_data(positions_array[:, 0], positions_array[:, 1])

        # B. Handle people who have disappeared
        pids_to_remove = []
        for pid, data in person_data.items():
            if pid not in pids_in_current_frame:
                data['missing_frames'] += 1
                data['box'].set_visible(False)
                data['text'].set_visible(False)
                
                if data['missing_frames'] > MISSING_THRESHOLD:
                    pids_to_remove.append(pid)
        
        # C. Clean up people who have been gone for too long
        for pid in pids_to_remove:
            person_data[pid]['box'].remove()
            person_data[pid]['line'].remove()
            person_data[pid]['text'].remove()
            del person_data[pid]
            
        return []

    # 4. Create the animation object
    print("Preparing animation...")
    anim = FuncAnimation(fig, update, frames=frame_ids, interval=1000/FPS, blit=False)
    
    # 5. Decide whether to show live or save to file
    if show_live:
        print("Showing live animation. Close the plot window to exit.")
        plt.show()
    else:
        try:
            print(f"Saving animation to '{OUTPUT_FILENAME}'... This may take a while.")
            anim.save(OUTPUT_FILENAME, writer='ffmpeg', progress_callback=lambda i, n: print(f'  -> Processing frame {i+1} of {n}', end='\r'))
            print("\nAnimation saved successfully!")
        except FileNotFoundError:
            print("\n\nError: ffmpeg not found. Could not save animation.")
            print("Please install ffmpeg or run with the '--show-live' flag to view interactively.")
            sys.exit(1)


if __name__ == '__main__':
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Visualize pedestrian trajectories from ETH/UCY datasets."
    )
    parser.add_argument(
        '--show-live',
        action='store_true',
        help="Display the animation in a live window instead of saving to a file."
    )
    args = parser.parse_args()

    # Call the main function with the user's choice
    create_trajectory_animation(show_live=args.show_live)