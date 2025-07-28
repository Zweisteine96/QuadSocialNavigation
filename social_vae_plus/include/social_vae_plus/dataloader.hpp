#ifndef SOCIAL_VAE_DATALOADER_HPP
#define SOCIAL_VAE_DATALOADER_HPP

#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>
#include <SFML/Graphics/Rect.hpp>

// A simpe struct to hold a single row of data from the original txt file
struct TrajectoryPoint {
    int frame_id;
    double person_id;
    double x;
    double y;
};

// A class to handle loading and processing the dataset
class DataLoader {
public:
    // Constructor that takes the path to the dataset file and delta_t
    DataLoader(const std::string& file_path, double delta_t);

    //NOTE: old method for getting history and neighbors for an agent at a specific frame.
    std::pair<torch::Tensor, torch::Tensor> get_history_and_neighbors(
        int person_id,
        int frame_id,
        int history_length,
        double neighbor_radius
    );

    // NOTE: new method to get only the history for a single agent at a specific frame.
    // Return an empty tensor if the history is not valid.
    torch::Tensor get_history(int person_id, int frame_id, int history_length);

    // Helper method to get all unique frame IDs
    std::vector<int> get_unique_frame_ids() const;

    // Helper method to get all persons present in a specific frame
    std::vector<int> get_persons_in_frame(int frame_id) const;

    sf::FloatRect get_dataset_bounds() const;

private:
    // Read the raw txt file
    void load_raw_data(float& max_x, float& max_y);

    // Organize raw data into per-person trajectories
    void process_into_trajectories();

    // Calculate velocities and accelerations for all trajectories
    void calculate_derivatives();

    std::string file_path_;
    double delta_t_;

    // Raw data as read from the file
    std::vector<TrajectoryPoint> raw_data_;

    // Organized data per person:
    // Key: person_id, Value: a map of <frame_id, features (pos, vel, acc)>
    // The inner tensor is of shape [6] for {x, y, vx, vy, ax, ay}
    std::map<int, std::map<int, torch::Tensor>> processed_trajectories_;

    // A map to quickly find all persons present in a given frame
    // Key: frame_id, Value: a vector of person_ids
    std::map<int, std::vector<int>> frame_to_persons_;

    // Sorted list of all unique frame IDs
    std::vector<int> unique_frame_ids_;

    sf::FloatRect dataset_bounds_; 
};


#endif