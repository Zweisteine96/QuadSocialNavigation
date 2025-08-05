import torch
import random
import numpy as np 

def ADE_FDE(y_, y, batch_first=False):
    # average displacement error
    # final displacement error
    # y_, y: S x L x N x 2
    if torch.is_tensor(y):
        err = (y_ - y).norm(dim=-1)
    else:
        err = np.linalg.norm(np.subtract(y_, y), axis=-1)
    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
    return ade, fde

def kmeans(k, data, iters=None):
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]

    if iters is None: iters = 100000
    for _ in range(iters):
    # while True:
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        closest = np.argmin(distances, axis=0)
        centroids_ = []
        for k in range(len(centroids)):
            cand = data[closest==k]
            if len(cand) > 0:
                centroids_.append(cand.mean(axis=0))
            else:
                centroids_.append(data[np.random.randint(len(data))])
        centroids_ = np.array(centroids_)
        if np.linalg.norm(centroids_ - centroids) < 0.0001:
            break
        centroids = centroids_
    return centroids

def FPC(y, n_samples):
    # y: S x L x 2
    goal = y[...,-1,:2]
    goal_ = kmeans(n_samples, goal)
    dist = np.linalg.norm(goal_[:,np.newaxis,:2] - goal[np.newaxis,:,:2], axis=-1)
    chosen = np.argmin(dist, axis=1)
    return chosen
    
def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])

# (Place this after your imports)

def inspect_dataloader(dataloader, config, name="Datalaloader"):
    """
    Prints a detailed report of a DataLoader's properties and the data it yields.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to inspect.
        config (module): The configuration module with parameters like OB_HORIZON.
        name (str, optional): A descriptive name for the dataloader for printing.
    """
    print("\n" + "="*60)
    print(f"  INSPECTING DATALOADER: {name.upper()}")
    print("="*60)

    # 1. Get the number of batches (iterations)
    try:
        # This works for custom samplers that have a __len__ method
        num_batches = len(dataloader.batch_sampler)
    except (AttributeError, TypeError):
        # Fallback for standard dataloaders
        num_batches = len(dataloader)

    if num_batches == 0:
        print("This dataloader is empty (contains 0 batches).")
        print("="*60 + "\n")
        return

    # 2. Get the total number of items in the dataset
    total_items = len(dataloader.dataset)

    # The batch size from config is a target. The actual size varies per batch.
    target_batch_size = config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 'N/A'

    print(f"Total items in underlying dataset: {total_items}")
    print(f"Number of batches (iterations): {num_batches}")
    print(f"Target batch size from config: {target_batch_size}")
    
    # 3. Inspect the first batch
    print("\n--- Inspecting the first batch ---")
    try:
        first_batch = next(iter(dataloader))
    except StopIteration:
        print("Could not fetch a batch; the dataloader is empty.")
        print("="*60 + "\n")
        return

    x, y, neighbor = first_batch

    print("\n[1] History Tensor (x):")
    print(f"  - Shape: {x.shape}")
    print(f"  - DType: {x.dtype}")
    print(f"  - Meaning: (Observation Horizon, Num Peds in Batch, Features)")
    print(f"             ({config.OB_HORIZON}, {x.shape[1]}, {x.shape[2]})")

    print("\n[2] Future Ground Truth Tensor (y):")
    print(f"  - Shape: {y.shape}")
    print(f"  - DType: {y.dtype}")
    print(f"  - Meaning: (Prediction Horizon, Num Peds in Batch, Features)")
    print(f"             ({config.PRED_HORIZON}, {y.shape[1]}, {y.shape[2]})")

    print("\n[3] Neighbor/Social Data:")
    if torch.is_tensor(neighbor):
        print(f"  - Type: torch.Tensor")
        print(f"  - Shape: {neighbor.shape}")
    else:
        print(f"  - Type: {type(neighbor)}")
        if isinstance(neighbor, list) and neighbor:
            print(f"  - Length: {len(neighbor)}")
            print(f"  - Type of first element: {type(neighbor[0])}")

    print("\n" + "="*60 + "\n")