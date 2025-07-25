#include "social_vae_plus/predictor.hpp"
#include <iostream>


Predictor::Predictor(const std::string& model_path) {
    try {
        // Load the model from the file
        module_ = torch::jit::load(model_path);
        
        // Move to CPU and set to evaluation mode
        module_.to(at::kCPU);
        module_.eval();

        is_initialized_ = true;
        std::cout << "--> Predictor initialized with model: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading mode in Predictor: " << e.what() << std::endl;
        is_initialized_ = false;
    }
}

// Implementation of the predict method
torch::Tensor Predictor::predict(const torch::Tensor& history, const torch::Tensor& neighbors) {
    if (!is_initialized_) {
        std::cerr << "Error: Predictor is not initialized." << std::endl;
        return torch::Tensor(); // Return an empty tensor on failure
    }

    // Package inputs into a vector of IValue 
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(history);
    inputs.push_back(neighbors);

    torch::Tensor output;
    try {
        // Run inference without calculating gradients for performance
        torch::NoGradGuard no_grad;
        output = module_.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Error during model inference: " << e.what() << std::endl;
        return torch::Tensor(); // Return an empty tensor on failure
    }
    return output;
}