#include "social_vae_plus/predictor.hpp"
#include "social_vae_plus/dataloader.hpp"
#include "social_vae_plus/animator.hpp"
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./prediction_app <path-to-model.pt> <path-to-data.txt> \n" << std::endl;
        return -1;
    }

    // Create dummy input data with correct shapes
    const int OB_HORIZON = 8;
    const int PRED_HORIZON = 12;
    const int BATCH_SIZE = 1;
    const int FEATURE_DIM = 6;
    const int NUM_NEIGHBORS = 5;
    const double DELTA_T = 0.4;
    const double NEIGHBOR_RADIUS = 2.0;
    bool USE_DUMMY_DATA = false;
    bool SHOW_ANIMATION = true;


    // Instantiate the Predictor and DataLoader
    Predictor predictor(argv[1]);
    DataLoader data_loader(argv[2], DELTA_T);

    if (USE_DUMMY_DATA) {
        std::cout << "--> Creat dummy input tensors for prediction test.\n";
        torch::Tensor history = torch::randn({OB_HORIZON, BATCH_SIZE, FEATURE_DIM});
        torch::Tensor neighbors = torch::randn({OB_HORIZON, BATCH_SIZE, NUM_NEIGHBORS, FEATURE_DIM});
        std::cout << "History tensor shape: " << history.sizes() << std::endl;
        std::cout << "Neighbors tensor shape: " << neighbors.sizes() << std::endl;

        // Call the predict model
        torch::Tensor output = predictor.predict(history, neighbors);
        std::cout << "Predicted output tensor shape: " << output.sizes() << std::endl;

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
    } else {
        if (!SHOW_ANIMATION) {
            std::cout << "--> Starting Full Dataset Prediction Loop. \n";
            auto all_frames = data_loader.get_unique_frame_ids();

            // Loop through every frame in the dataset that could be a prediction point
            // We start from OB_HORIZON - 1 becasue we need a full history window
            for (size_t frame_idx = OB_HORIZON - 1; frame_idx < all_frames.size(); ++frame_idx) {
                int current_frame_id = all_frames[frame_idx];
                // Gather agents for batch prediction
                std::vector<int> pids_for_prediction;
                std::vector<torch::Tensor> history_batch_list;

                auto persons_in_frame = data_loader.get_persons_in_frame(current_frame_id);
                for (int person_id : persons_in_frame) {
                torch::Tensor history = data_loader.get_history(person_id, current_frame_id, OB_HORIZON);
                if (history.numel() > 0) {
                    pids_for_prediction.push_back(person_id);
                    history_batch_list.push_back(history);
                }
                }
                // If no agents have valid history at this frame, continue to the next frame
                if (pids_for_prediction.empty()) {
                continue;
                }
                std::cout << "\n--- Frame " << current_frame_id << ": Preidcting for " << pids_for_prediction.size() << " agents ---\n" << std::endl;

                // Construct batch tensors for history and neighbors
                torch::Tensor history_batch = torch::stack(history_batch_list, 1); // History tensor shape: [seq_len, batch_size, feature_dim]
                
                // Consider all other agents as neighbors
                int n_agents = pids_for_prediction.size();
                std::vector<torch::Tensor> neighbor_batch_list;
                for (int i = 0; i < n_agents; ++i) {
                // For the i-th agent in the group, its neighbors are all other agents in the same group
                std::vector<torch::Tensor> neighbor_tensors;
                for (int j = 0; j < n_agents; ++j) {
                    if (i == j) continue;
                    neighbor_tensors.push_back(history_batch.select(1, j)); // Select the j-th agent's history
                }

                if (neighbor_tensors.empty()) {
                    // If there is only one agent without any neighbors,
                    // we provide a placeholder tensor of shape [seq_len, 1, feature_dim]
                    neighbor_batch_list.push_back(torch::full({OB_HORIZON, 1, 6}, 1e9));
                } else {
                    // Stack along the num_neighbors dimension
                    neighbor_batch_list.push_back(torch::stack(neighbor_tensors, 1));
                }
                }

                // Pad neighbors to the same size before stacking into a single batch tensor
                // Find the max number of neighbors in this group.
                // NOTE: here we don't really need this padding, since we adopt "all other agents as neighbors" logic.
                int max_neighbors = 0;
                for (const auto& n : neighbor_batch_list) {
                if (n.size(1) > max_neighbors) {
                    max_neighbors = n.size(1);
                }
                }

                for (auto& n : neighbor_batch_list) {
                int current_neighbors = n.size(1);
                int padding_needed = max_neighbors - current_neighbors;
                if (padding_needed > 0) {
                    torch::Tensor padding = torch::full({OB_HORIZON, padding_needed, 6}, 1e9);
                    n = torch::cat({n, padding}, 1);
                }
                }
                torch::Tensor neighbor_batch = torch::stack(neighbor_batch_list, 1);

                // RUn batched inference
                auto start_time = std::chrono::high_resolution_clock::now();
                
                torch::Tensor prediction_batch = predictor.predict(history_batch, neighbor_batch);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

                if (prediction_batch.numel() == 0) {
                    std::cerr << "  Prediction failed for this batch. Skipping." << std::endl;
                    continue;
                }

                std::cout << "  Inference successful in " << duration.count() << " microseconds." << std::endl;

                // Process and display prediction results
                // Prediction tensor shape: [pred_horizon, batch_size, 2]
                for (int i = 0; i < n_agents; ++i) {
                    int person_id = pids_for_prediction[i];
                    torch::Tensor single_prediction = prediction_batch.select(1, i);

                    // Print only the first and last predicted points
                    float start_x = single_prediction[0][0].item<float>();
                    float start_y = single_prediction[0][1].item<float>();
                    float end_x = single_prediction[PRED_HORIZON - 1][0].item<float>();
                    float end_y = single_prediction[PRED_HORIZON - 1][1].item<float>();

                    std::cout << "  -> Agent " << person_id << ": Predicted path starts at ("
                            << start_x << ", " << start_y << ") and ends at ("
                            << end_x << ", " << end_y << ")." << std::endl;
                }
            }
            std::cout << "\n--- Finished processing all frames. ---\n";
            return 0;
        } else {
            sf::FloatRect bounds = data_loader.get_dataset_bounds();
            Animator animator("SocialVAE C++ Prediction", 14, 12, bounds);
            animator.run(data_loader, predictor);

            std::cout << "\n--- Animation finished. ---\n";
            return 0;
        }
    }
}