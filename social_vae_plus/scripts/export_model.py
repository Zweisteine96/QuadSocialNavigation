r"""Export the model to TrochScript format that C++ can use."""
import torch
import argparse
import sys
from pathlib import Path
from social_vae import SocialVAE

# --- CONFIGURATION --- #
OB_HORIZON = 8
PRED_HORIZON = 12
DELTA_T = 0.4
OB_RADIUS = 2
RNN_HIDDEN_DIM = 256


def export_model(ckpt_path, export_path):
    """Export the model to TorchScript format."""
    print(f"--> Loading source model from '{ckpt_path}'...")
    device = torch.device('cpu') # right now cpu-only machine

    # Load model architeture and state dict
    try:
        model = SocialVAE(
            horizon=OB_HORIZON,
            ob_radius=OB_RADIUS,
            hidden_dim=RNN_HIDDEN_DIM
        )
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create dummy inputs with the correct shapes for tracing
    batch_size = 1
    dummy_x = torch.randn(OB_HORIZON, batch_size, 6, device=device)
    dummy_neighbots = torch.randn(OB_HORIZON, batch_size, 5, 6, device=device)

    # Create a wrapper 
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, neighbors):
            # Hardcode n_predictions=1 and select first sample from the output list
            return self.model(x, neighbors, n_predictions=1)[0]
    wrapper = ModelWrapper(model)
    wrapper.eval()

    # Trace the model
    print(f"--> Tracing model...")
    traced_script_module = torch.jit.trace(wrapper, (dummy_x, dummy_neighbots))

    # Save the traced model
    traced_script_module.save(export_path)
    print(f"--> Model successfully exported to '{export_path}'.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_ckpt', required=True, type=str,
        help='Path to load the original trained model checkpoint.'
    )
    parser.add_argument(
        '--export_path', default='social_vae_traced.pt', type=str,
        help='Path to save the traced model.'
    )
    args = parser.parse_args()

    export_model(args.model_ckpt, args.export_path)


