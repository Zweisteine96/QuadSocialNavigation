#include "social_vae_plus/dataloader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Constructor: Initialize members and call the processing pipeline
DataLoader::DataLoader(const std::string& file_path, double delta_t)
    : file_path_(file_path), delta_t_(delta_t) {
    // Initialize bounds to be inverted so the first point will set the correctly
    dataset_bounds_.left = std::numeric_limits<float>::max();
    dataset_bounds_.top = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();

    std::cout << "--> Initializing DataLoader with file: " << file_path_ << std::endl;

    load_raw_data(max_x, max_y); // Load the raw data from the file
    dataset_bounds_.width = max_x - dataset_bounds_.left;
    dataset_bounds_.height = max_y - dataset_bounds_.top;

    std::cout << "  - Dataset bounds: min(" << dataset_bounds_.left << ", " << dataset_bounds_.top 
              << ") max(" << max_x << ", " << max_y << ")" << std::endl;

    process_into_trajectories(); // Organize the data into per-person trajectories
    calculate_derivatives(); // Calculate velocities and accelerations
    std::cout << "--> DataLoader initialized successfully." << std::endl;
}

// Load data from the txt file
void DataLoader::load_raw_data(float& max_x, float& max_y) {
    std::ifstream file(file_path_);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file " << file_path_ << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        TrajectoryPoint point;
        ss >> point.frame_id >> point.person_id >> point.x >> point.y;
        // Check if the stream is still valid
        if (ss) {
            raw_data_.push_back(point);

            // --- UPDATE BOUNDS ---
            if (point.x < dataset_bounds_.left) dataset_bounds_.left = point.x;
            if (point.y < dataset_bounds_.top) dataset_bounds_.top = point.y;
            if (point.x > max_x) max_x = point.x;
            if (point.y > max_y) max_y = point.y;
        } else {
            std::cerr << "Warning: Invalid line in data file: " << line << std::endl;
        }
    }
    std::cout << " - Loaded " << raw_data_.size() << " raw data points." << std::endl;
}

sf::FloatRect DataLoader::get_dataset_bounds() const {
    return dataset_bounds_;
}

//Organize the flat list of points into structured trajectories
void DataLoader::process_into_trajectories() {
    std::map<int, bool> seen_frames;

    for (const auto& point: raw_data_) {
        // Create a 6-element tensor for {x, y, vx, vy, ax, ay} , initialized to NaN
        torch::Tensor features = torch::full({6}, std::numeric_limits<double>::quiet_NaN());
        features[0] = point.x; // x position
        features[1] = point.y; // y position

        processed_trajectories_[point.person_id][point.frame_id] = features;
        frame_to_persons_[point.frame_id].push_back(point.person_id);

        if (seen_frames.find(point.frame_id) == seen_frames.end()) {
            // This searches the seen_frames map for the key point.frame_id
            // If the key is not found, .find() returns a special iterator that is equal to end()
            unique_frame_ids_.push_back(point.frame_id);
            seen_frames[point.frame_id] = true; // Mark this frame as seen
        }
    }
    std::sort(unique_frame_ids_.begin(), unique_frame_ids_.end());
    std::cout << " - Processed data for " << processed_trajectories_.size() << " unique persons across "
              << unique_frame_ids_.size() << " unique frames." << std::endl;
}

// Calculate derivatives (velocity and acceleration) for each trajectory
void DataLoader::calculate_derivatives() {
    // Iterate over each person in our map
    for (auto& person_pair: processed_trajectories_) {
        auto& trajectory = person_pair.second; // This is the map <frame_id, features_tensor>

        // We can't use iterators directly for derivatives,
        // so we'll get the sorted frames
        std::vector<int> frames;
        for (const auto& frame_pair: trajectory) {
            frames.push_back(frame_pair.first);
        }
        std::sort(frames.begin(), frames.end());

        if (frames.size() < 3) continue; // Need at least 3 points for acc derivatives

        // Convert to position tensor for easier calculation
        std::vector<torch::Tensor> positions;
        for (int frame: frames) {
            positions.push_back(trajectory.at(frame).slice(0, 0, 2));
        }
        torch::Tensor pos_tensor = torch::stack(positions);

        // Calculate velocities and accelerations using torch.gradient
        // NOTE: torch.gradient needs a coordinate spacing. 
        // The difference between frames is not always 1, so we build a tensor of frame IDs.
        torch::Tensor frame_coords = torch::tensor(frames, torch::kInt64);
        torch::Tensor time_coords = torch::tensor(frames, torch::kFloat64) * delta_t_;

        auto vel_list = torch::gradient(
            pos_tensor, /*spacing=*/{time_coords}, /*dim=*/at::IntArrayRef({0})
        );
        torch::Tensor vel_tensor = vel_list[0];
        
        auto acc_list = torch::gradient(
            vel_tensor, /*spacing=*/{time_coords}, /*dim=*/at::IntArrayRef({0})
        );
        torch::Tensor acc_tensor = acc_list[0];

        // Put the calculated velocities and accelerations into trajectory
        for (size_t i = 0; i < frames.size(); ++i) {
            int frame_id = frames[i];
            trajectory.at(frame_id).slice(0, 2, 4) = vel_tensor[i]; // Set vx, vy
            trajectory.at(frame_id).slice(0, 4, 6) = acc_tensor[i]; // Set ax, ay
        }
    }
    std::cout << " - Calculated velocities and accelerations for all trajectories." << std::endl;
}

std::vector<int> DataLoader::get_unique_frame_ids() const {
    return unique_frame_ids_;
}

std::vector<int> DataLoader::get_persons_in_frame(int frame_id) const {
    if (frame_to_persons_.count(frame_id)) {
        return frame_to_persons_.at(frame_id);
    }
    return {}; // Return empty vector if frame not found
}

// New method to get only the single agent's history for the prediction model
// The neighbor logic will be handled in main.cpp where we have the context of the entire batch.
torch::Tensor DataLoader::get_history(int person_id, int current_frame_idx, int history_length) {
    // Safety check
    if (current_frame_idx < history_length - 1) {
        return torch::Tensor();
    }
    std::vector<torch::Tensor> history_features;

    // Here assume constant frame steps.
    // A more robust way is to search for the closest avaiable frame ID.
    //int frame_step = unique_frame_ids_[1] - unique_frame_ids_[0];

    for (int i = 0; i < history_length; ++i) {
        //int frame_to_get = current_frame_id - (history_length - 1 - i) * frame_step;
        int frame_idx_to_get = current_frame_idx - (history_length - 1 - i);
        int frame_id_to_get = unique_frame_ids_[frame_idx_to_get];

        // Check if the person exists and has data for that specific frame
        auto person_it = processed_trajectories_.find(person_id); 
        if (person_it != processed_trajectories_.end()) {
            // person_it now points to both the label (person_id) and the contents
            auto frame_it = person_it->second.find(frame_id_to_get);
            if (frame_it != person_it->second.end()) {
                // Check if the data is valid
                if (!torch::any(torch::isnan(frame_it->second)).item<bool>()) {
                    history_features.push_back(frame_it->second);
                } else {
                    return torch::Tensor(); // Return empty tensor if data is invalid
                }
            }
        } else {
            return torch::Tensor(); // Return empty tensor if person_id not found
        }
    }

    // If we collected the full history, stack it into a single tensor
    if (history_features.size() != static_cast<size_t>(history_length)) {
        return torch::Tensor(); // Return empty tensor if history is incomplete
    }
    return torch::stack(history_features); // Shape [history_length, 6]
}



// (NOT USED!) Main public interface to get data for the prediction model
std::pair<torch::Tensor, torch::Tensor> DataLoader::get_history_and_neighbors(
    int person_id, int current_frame_id, int history_length, double neighbor_radius
) {
    std::vector<torch::Tensor> history_features;
    bool history_valid = true;
    int constant_frame_step = unique_frame_ids_[1] - unique_frame_ids_[0];

    // 1. Get history for the target agent
    for (int i = 0; i < history_length; ++i) {
        int frame_to_get = current_frame_id - (history_length - 1 - i) * static_cast<int>(constant_frame_step);

        if (processed_trajectories_.count(person_id) && processed_trajectories_.at(person_id).count(frame_to_get)) {
            history_features.push_back(processed_trajectories_.at(person_id).at(frame_to_get));
        } else {
            history_valid = false; // If any frame is missing, we invalidate the history
            break;
        }
    }

    if (!history_valid || history_features.size() != static_cast<size_t>(history_length)) {
        // Return empty tensors if history is invalid
        return {torch::Tensor(), torch::Tensor()};
    }

    torch::Tensor history_tensor = torch::stack(history_features); // Shape [history_length, 6]

    // 2. Get neighbors for each step in the history
    std::vector<torch::Tensor> all_neighbors_over_time;

    for (int i = 0; i < history_length; ++i) {
        int frame_to_get = current_frame_id - (history_length - 1 - i) * static_cast<int>(constant_frame_step);
        torch::Tensor agent_pos = history_tensor[i].slice(0, 0, 2); // Get agent's position at this frame

        std::vector<torch::Tensor> neighbors_in_frame;
        // Find all other people in that frame
        if (frame_to_persons_.count(frame_to_get)) {
            for (int other_person_id : frame_to_persons_.at(frame_to_get)) {
                if (other_person_id == person_id) continue; // Skip the agent itself

                torch::Tensor other_pos = processed_trajectories_.at(other_person_id).at(frame_to_get).slice(0, 0, 2);
                if (torch::dist(agent_pos, other_pos).item<double>() < neighbor_radius) {
                    // If the distance is within the neighbor radius, add to neighbors
                    neighbors_in_frame.push_back(processed_trajectories_.at(other_person_id).at(frame_to_get));
                }
            }
        }

        if (neighbors_in_frame.empty()) {
            // If no neighbors found, add a tensor of NaNs or a special value.
            // Model expects a tensor, even if it's empty of real neighbors.
            // Create a placeholder of shape [1, 6].
            all_neighbors_over_time.push_back(torch::full({1, 6}, 1e9));
        } else {
            all_neighbors_over_time.push_back(torch::stack(neighbors_in_frame));
        }
    }

    torch::Tensor neighbor_tensor = torch::full({history_length, 1, 6}, 1e9); //Placeholder

    // Reshape for batch size of 1
    history_tensor = history_tensor.unsqueeze(1); // Shape [seq_length, batch_size, 6]
    neighbor_tensor = neighbor_tensor.unsqueeze(1); // Placeholder shape [seq_length, batch_size, n_neighbors, 6]

    return {history_tensor, neighbor_tensor};
}