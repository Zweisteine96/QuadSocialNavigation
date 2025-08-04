// Copyright 2025 CHENG GUO
// Licensed under the Apache License, Version 2.0

// This file is simply to inspect the nested JSON messages.
// It reads the FIRST message from the sensor_sim/Obstacles topic,
// then perfomrs a two-level JSON parse,
// and prints the key-value strcutures of the INNER JSON object.

// An example of the running result is shown below:
// [INFO] [1754040347.593531322] [rosbag2_storage]: Opened database 'src/social_pipeline_node/data/my_bag/my_bag_0.db3' for READ_ONLY.
// Inspecting first message on topic: /sensor_sim/Obstacles
// ----------------------------------------------------
// Found a message. Performing nested JSN parse...
//
// Inspecting inner data for the first obstacle found: "BP_CrownRandom1_C_1"
//
// Discovered Key-Value Pairs:
// ---------------------------
// {
//   "name": "BP_CrownRandom1_C_1",
//   "vx": -1.1865552677421973,
//   "vy": -1.2751774131709825,
//   "x": 1.1250105259982135,
//   "y": 33.71926187686979,
//   "z": 0.9215000152587891
// }
// ---------------------------
// Inspection complete. Exiting.


#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <iomanip>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_storage/storage_options.hpp"
#include "example_interfaces/msg/string.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char *argv[]) 
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "<path-to-bag-directory>" << std::endl;
        return 1;
    }

    const std::string bag_path = argv[1];
    const std::string topic_to_inspect = "/sensor_sim/Obstacles";
    const std::string output_filename = "json_message_example.yaml";

    // Create an output file stream
    std::ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_filename << std::endl;
        return 1;
    }

    rosbag2_cpp::Reader reader;
    try{
        //reader.open({bag_path, "sqlite3"});
        rosbag2_storage::StorageOptions storage_options({bag_path, "sqlite3"});
        reader.open(storage_options);
    } catch (const std::exception & ex) {
        std::cerr << "Error opening bag file: " << ex.what() <<std::endl;
        return 1;
    }

    rclcpp::Serialization<example_interfaces::msg::String> serialization;

    //std::cout << "Inspecting first message on topic: " << topic_to_inspect << std::endl;
    //std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Found a message. Analyzing and writing content to " << output_filename << "..." << std::endl;

    while (reader.has_next()) {
        auto serialized_message_from_bag = reader.read_next();

        if (serialized_message_from_bag->topic_name == topic_to_inspect) {
            auto ros_message = std::make_shared<example_interfaces::msg::String>();

            // Fix: create th etype of serialized message that the rclcpp deserializer expectes
            // Initialize it using the raw data buffer from the message we read from the bag
            rclcpp::SerializedMessage rcl_serialized_message(*serialized_message_from_bag->serialized_data);

            // Pass a pointer to the just created object
            serialization.deserialize_message(&rcl_serialized_message, ros_message.get());

            std::cout << "Found a message. Parsing its full content...\n" << std::endl;

            try {
                // Handle the timestamp from the bag message
                long long timestamp_ns = serialized_message_from_bag->time_stamp;
                double timestamp_sec = static_cast<double>(timestamp_ns) / 1e9;
                // First parse the main string from the raw ROS2 message
                json outer_json = json::parse(ros_message->data);

                if (outer_json.is_object() && !outer_json.empty()) {
                    // // Take the very first key-value pair from the outer JSON map
                    // auto first_item = outer_json.begin();
                    // std::string obstacle_key = first_item.key();
                    // std::string inner_json_string = first_item.value().get<std::string>();

                    // std::cout << "Inspecting inner data for the first obstacle found: \"" << obstacle_key << "\"\n\n";

                    // try {
                    //     // Then try to parse the inner JSON string
                    //     json inner_json = json::parse(inner_json_string);

                    //     // Then print the keys and their value types from the inner JSON
                    //     std::cout << "Discovered Key-Value Pairs:\n";
                    //     std::cout << "---------------------------\n";

                    //     std::cout << inner_json.dump(2) << std::endl;
                    //     std::cout << "---------------------------\n";
                    // } catch (json::parse_error& e_inner) {
                    //     std::cerr << "Inner JSON parse error for key '" << obstacle_key << "': " << e_inner.what() << '\n';
                    // }
                    
                    // FIX: loop through all items in the outer map
                    int crown_count = 0;
                    int pillar_count = 0;
                    std::vector<std::string> crown_names;
                    std::vector<std::string> pillar_names;
                    json special_item_data;
                    bool special_item_found = false;

                    for (auto it = outer_json.begin(); it != outer_json.end(); ++it) {
                        const std::string& key = it.key();
                        if (key.find("BP_CrownRandom_C_") == 0) {
                            crown_count++;
                            crown_names.push_back(key);
                        } else if (key.find("BP_Pillar_C_") == 0) {
                            pillar_count++;
                            pillar_names.push_back(key);
                        }

                        if (key == "BP_CrownRandom1_C_1") {
                            special_item_found = true;
                            try {
                                //special_item_data = json::parse(it.value().dump()).get<std::string>();
                                std::string inner_json_str = it.value().get<std::string>();
                                special_item_data = json::parse(inner_json_str);
                            } catch(...) {special_item_found = false;}
                        }
                    }

                    output_file << "summary:\n";
                    output_file << "  timestamp_ns: " << timestamp_ns << "\n";
                    output_file << "  timestamp_sec: " << std::fixed << std::setprecision(9) << timestamp_sec << "\n";
                    output_file << "  total_obstacles_in_frame: " << outer_json.size() << "\n";
                    output_file << "  crown_type_count: " << crown_count << "\n";
                    output_file << "  pillar_type_count: " << pillar_count << "\n";
                    output_file << "\n";

                    if (special_item_found) {
                        output_file << "special_item_details:\n";
                        output_file << "  name: BP_CrownRandom1_C_1\n";
                        if (special_item_data.is_object()) {
                            for (auto inner_it = special_item_data.begin(); inner_it != special_item_data.end(); ++inner_it) {
                                output_file << "  " << inner_it.key() << ": " << inner_it.value().dump() << "\n";
                            }
                        } else {
                            output_file << "  data: " << special_item_data.dump() << "\n";
                        }
                        output_file << "\n";
                    }

                    // Write name list to the yaml file
                    output_file << "crown_obstacle_names:\n";
                    for (const auto& name : crown_names) {
                        output_file << "  - " << name << "\n";
                    }
                    output_file << "\n";
                    output_file << "pillar_obstacle_names:\n";
                    for (const auto& name : pillar_names) {
                        output_file << "  - " << name << "\n";
                    }
                    output_file << "\n";

                    output_file << "all_obstacles_detailed:\n";
                    for (auto it = outer_json.begin(); it != outer_json.end(); ++it) {
                        //std::string obstacle_key = it.key();
                        // Write the top-level key (the obstacle name) to the file
                        output_file << " " << it.key() << ":\n";
                        //std::cout << "--- Obstacle: \"" << obstacle_key << "\" ---\n";

                        try {
                            std::string inner_json_string = it.value().get<std::string>();
                            json inner_json = json::parse(inner_json_string);

                            // Iterate through the inner key-value pairs to write them in YAML format
                            if (inner_json.is_object()) {
                                for(auto inner_it = inner_json.begin(); inner_it != inner_json.end(); ++inner_it) {
                                    output_file << "    " << inner_it.key() << ": " << inner_it.value().dump() << "\n";
                                }
                            } else {
                                // If it's not an object (e.g., an array), just dump the raw value.
                                output_file << "    data: " << inner_json.dump() << "\n";
                            }
                        } catch (const std::exception& e_inner) {
                            //std::cerr << " Error parsing inner JSON for this obstacle: " << e_inner.what() << '\n' << std::endl;
                            output_file << "  error: \"Failed to parse inner JSON: " << e_inner.what() << "\"\n\n";
                        }
                    }
                } else {
                    //std::cerr << "The outer JSON object is empty or not an object." << std::endl;
                    output_file << "error: The outer JSON object is empty or not an object.\n";
                }
            } catch (json::parse_error& e_outer) {
                //std::cerr << "Outer JSON parse error: " << e_outer.what() << '\n';
                output_file << "error: Outer JSON parse error: " << e_outer.what() << "\n";
            }

            // We take only the first message
            break;
        }
    }
    output_file.close();
    //std::cout << "----------------------------------------------------" << std::endl;
    //std::cout << "Inspection complete. Displayed all obstacles from the first frame." << std::endl;
    std::cout << "Inspection complete. Results saved to " << output_filename << std::endl;
    return 0;
}
