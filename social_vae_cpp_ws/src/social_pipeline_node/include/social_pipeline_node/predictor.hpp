#ifndef SOCIAL_VAE_PREDICTOR_HPP
#define SOCIAL_VAE_PREDICTOR_HPP

#include <torch/script.h>
#include <string>

// This class encapsulates all logic for loading and running the model
class Predictor {
public:
    // Constructor that loads the model from a file path
    explicit Predictor(const std::string& model_path);
    
    // Run a prediction given history and neighbor data
    torch::Tensor predict(const torch::Tensor& history, const torch::Tensor& neighbors);

    bool is_initialized() const { return is_initialized_; }

private:
    torch::jit::script::Module module_; // The loaded TorchScript model
    bool is_initialized_ = false; // Flag to ensure the model was loaded correctly
};

#endif // SOCIAL_VAE_PREDICTOR_HPP
