#include "social_vae_plus/predictor.hpp"
#include <torch/script.h>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./prediction_app models/exported/eth/social_vae_plus.pt" << std::endl;
        return -1;
    }

    // Instantiate the Predictor, which loads the model
    Predictor predictor(argv[1]);

    // Create dummy input data with correct shapes
    const int OB_HORIZON = 8;
    const int BATCH_SIZE = 1;
    const int FEATURE_DIM = 6;
    const int NUM_NEIGHBORS = 5;

    torch::Tensor history = torch::randn({OB_HORIZON, BATCH_SIZE, FEATURE_DIM});
    torch::Tensor neighbors = torch::randn({OB_HORIZON, BATCH_SIZE, NUM_NEIGHBORS, FEATURE_DIM});
    std::cout << "--> Created dummy input tensors for prediction test.\n";

    // Call the predict model
    torch::Tensor output = predictor.predict(history, neighbors);

    // Check for a valid result and print it
    if (output.numel() == 0) {
        std::cerr << "Prediction failed. Existing." << std::endl;
        return -1;
    }

    std::cout << "\n--> Prediction Successful <--\n";
    output = output.squeeze(1); // Remove the batch dimension

    std::cout << "Predicted Trajectory (x, y) for 12 steps:\n";
    for (int i =0; i < output.size(0); ++i) {
        float x = output[i][0].item<float>();
        float y = output[i][1].item<float>();
        std::cout << " Step " << i+1 << ": (" << x << ", " << y << ")\n";
    }
    return 0;
}