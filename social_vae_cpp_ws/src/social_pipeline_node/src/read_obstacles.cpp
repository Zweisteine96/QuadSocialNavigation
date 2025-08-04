// Copyright 2025 CHENG GUO
// Licensed under the Apache License, Version 2.0

// This file reads the JSON-format message in the original .db3 file,
// retrieve the [x, v, vx, vy, ax, ay] data of each agent at each frame,
// then save them to a CSV file in a pre-defined format.

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <cmath>
#include <sstream>
//#include <iomanip>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_storage/storage_options.hpp"
//#include "rosbag2_storage/serialized_bag_message.hpp"
#include "example_interfaces/msg/string.hpp"

#include <nlohmann/json.hpp>
using json  =nlohmann::json;

// A struct to hold the complete state of an agent at a frame
struct AgentState {
    long long timestamp_ns = 0;
    double x = 0.0, y = 0.0;
    double vx = 0.0, vy = 0.0;
    double ax = 0.0, ay = 0.0;
    std::vector<std::string> neighbor_ids;
};

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Error: You should provide the path to the bag directory." << std::endl;
        std::cerr << "Usage: " << argv[0] << "<path_to_bag_directory>" << std::endl;
        return 1; // Exiting
    }

    // Configuration
    const std::string bag_path = argv[1];
    const std::string topic_to_extract = "/sensor_sim/Obstacles";
    const std::string output_filename = "unreal_pedestrian_data.csv";

    std::cout << "[DEBUG] Program started. Will read from: " << bag_path << std::endl;
    std::cout << "[DEBUG] Will look for topic: '" << topic_to_extract << "'" << std::endl;

    const double NEIGHBOR_DISTANCE_THRESHOLD = 2.0;

    // File setup
    std::ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_filename << std::endl;
        return 1;
    }
    
    // Define the header for the CSV file
    output_file << "timestamp_ns, obstacle_name, x, y, vx, vy, ax, ay, neighbor_ids" << std::endl;

    rosbag2_cpp::Reader reader;
    try {
        rosbag2_storage::StorageOptions storage_options({bag_path, "sqlite3"});
        reader.open(storage_options);
    } catch (const std::exception & ex) {
        std::cerr << "Error opening bag file: " << ex.what() << std::endl;
        return 1;
    }
    rclcpp::Serialization<example_interfaces::msg::String> serialization;

    // Store each agent's state from the previous frame
    std::map<std::string, AgentState> previous_frame_states;
    long long total_records_written = 0;
    int message_counter = 0;
    std::cout << "[DEBUG] Starting to loop through bag messages..." << std::endl;

    while (reader.has_next()) {
        message_counter++;
        auto serialized_message_from_bag = reader.read_next();

        if (message_counter < 100) { // Limit spam, print first 100 topics
             std::cout << "[DEBUG 1] Found message #" << message_counter 
                       << " on topic: '" << serialized_message_from_bag->topic_name << "'" << std::endl;
        }

        if (serialized_message_from_bag->topic_name == topic_to_extract) {
            std::cout << "[DEBUG 2] Matched target topic! Processing message." << std::endl;

            auto ros_message = std::make_shared<example_interfaces::msg::String>();
            rclcpp::SerializedMessage rcl_serialized_message(*serialized_message_from_bag->serialized_data);
            serialization.deserialize_message(&rcl_serialized_message, ros_message.get());

            try {
                std::cout << "[DEBUG 3] Attempting to parse outer JSON..." << std::endl;

                long long timestamp_ns = serialized_message_from_bag->time_stamp;
                json outer_json = json::parse(ros_message->data);

                std::cout << "[DEBUG 4] Outer JSON parsed successfully." << std::endl;
                std::map<std::string, AgentState> current_frame_states;

                if (outer_json.is_object()) {
                    std::cout << "[DEBUG 5] Looping through " << outer_json.size() << " items in outer JSON." << std::endl;

                    // 1. Populate current frame data to calculate acceleration
                    for (auto it = outer_json.begin(); it != outer_json.end(); ++it) {
                        const std::string& key = it.key();
                        std::cout << "[DEBUG 6] Found key: '" << key << "'" << std::endl;

                        if (key.find("BP_CrownRandom_C_") == 0) {
                            std::cout << "[DEBUG 7] Key '" << key << "' MATCHED THE FILTER!" << std::endl;
                            try {
                                json inner_json = json::parse(it.value().get<std::string>());
                                AgentState current_state;
                                current_state.timestamp_ns = timestamp_ns;
                                current_state.x = inner_json.value("x", 0.0);
                                current_state.y = inner_json.value("y", 0.0);
                                current_state.vx = inner_json.value("vx", 0.0);
                                current_state.vy = inner_json.value("vy", 0.0);

                                // Check memory for previous state to calculate accelerations
                                if (previous_frame_states.count(key)) {
                                    const auto& prev_state = previous_frame_states.at(key);
                                    double delta_t = (current_state.timestamp_ns - prev_state.timestamp_ns) / 1e9;

                                    if (delta_t > 1e-6) {
                                        current_state.ax = (current_state.vx - prev_state.vx) / delta_t;
                                        current_state.ay = (current_state.vy - prev_state.vy) / delta_t;
                                    }
                                }
                                current_frame_states[key] = current_state;
                            } catch (...) {
                                std::cerr << "Warning: An unknown, non-standard exception occurred. Skipping message." << std::endl;
                            }
                        }
                    }
                    // 2. Find neighbors for each agent in the current frame
                    for (auto& ego_pair : current_frame_states) {
                        for (const auto& other_pair : current_frame_states) {
                            if (ego_pair.first == other_pair.first) continue; // Skip the agent itself

                            double dx = ego_pair.second.x - other_pair.second.x;
                            double dy = ego_pair.second.y  -other_pair.second.y;
                            double distance = std::sqrt(dx * dx + dy * dy);

                            if (distance < NEIGHBOR_DISTANCE_THRESHOLD) {
                                ego_pair.second.neighbor_ids.push_back(other_pair.first);
                            }
                        }
                    }
                    // 3. Write the enriched data to the file
                    for (const auto& pair : current_frame_states) {
                        const AgentState& state = pair.second;

                        // Build the semicolcon-separated neighbor string
                        std::stringstream ss;
                        for (size_t i = 0; i < state.neighbor_ids.size(); ++i) {
                            ss << state.neighbor_ids[i] << (i == state.neighbor_ids.size() - 1 ? "" : ";");
                        }

                        output_file << state.timestamp_ns << "," << pair.first << ","
                                    << std::fixed << std::setprecision(6)
                                    << state.x << "," << state.y << "," << state.vx << "," << state.vy << ","
                                    << state.ax << "," << state.ay << "," << ss.str() << std::endl;
                        total_records_written++;
                    }
                }
                // Update memory
                previous_frame_states = current_frame_states;
            } catch (...) {
                std::cerr << "Warning: An unknown, non-standard exception occurred. Skipping message." << std::endl;
            }
        }
    }

    output_file.close();
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Feature Engineering Complete." << std::endl;
    std::cout << "Wrote " << total_records_written << " records to " << output_filename << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return 0;
}
