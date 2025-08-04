// Copyright 2025 CHENG GUO
// Licensed under the Apache License, Version 2.0

// This node subscribes to the raw JSON data from a ROS2 bag file, parses it,
// filters for pedestrians, and publishes the data 
// as a clean social_msgs::msg::AgentStateArray message once per frame.

#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <chrono>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "social_msgs/msg/agent_state_array.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_storage/storage_options.hpp"
#include "example_interfaces/msg/string.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class DataCleanerNode : public rclcpp::Node
{
public:
    DataCleanerNode(const std::string& bag_path)
    : Node("data_cleaner_node"), frame_id_counter_(0) {
        // Store the path to the bag file
        bag_path_ = bag_path;

        // Create a publisher that sends AgentStateArray messages on the "/raw_agents" topic
        publisher_ = this->create_publisher<social_msgs::msg::AgentStateArray>("/raw_agents", 10);

        RCLCPP_INFO(this->get_logger(), "Start streaming JSON-format messages from offline ROS2 bag data.");
        RCLCPP_INFO(this->get_logger(), "Read from bag data at: %s", bag_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing processed data on topic: '/raw_agents'");
    }

    void process_bag() {
        rosbag2_cpp::Reader reader;
        try{
            rosbag2_storage::StorageOptions storage_options({bag_path_, "sqlite3"});
            reader.open(storage_options);
        } catch (const std::exception & ex) {
            RCLCPP_ERROR(this->get_logger(), "Error opening bag file: %s", ex.what());
            rclcpp::shutdown();
            return;
        }

        rclcpp::Serialization<example_interfaces::msg::String> serialization;
        long long last_message_timestamp_ns = 0;
        bool first_message = true;
        
        // We iterate through every frame in the given bag file
        while (reader.has_next() && rclcpp::ok()) {
            // Get the current frame message
            auto serialized_message_from_bag = reader.read_next();

            if (!first_message) {
                auto message_timestamp = std::chrono::nanoseconds(serialized_message_from_bag->time_stamp);
                auto last_timestamp = std::chrono::nanoseconds(last_message_timestamp_ns);
                const double RATE_MULTIPLIER = 10.0;
                auto time_diff = message_timestamp - last_timestamp;
                if (time_diff.count() > 0) {
                    // Only sleep if time has advanced
                    //std::this_thread::sleep_for(time_diff);
                    // Divide the sleep duration by the multiplier to speed up playback
                    std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(time_diff / RATE_MULTIPLIER));
                }
            }
            last_message_timestamp_ns = serialized_message_from_bag->time_stamp;
            first_message = false;
            
            // We only read data from /sensor_sim/Obstacles topic
            if (serialized_message_from_bag->topic_name == "/sensor_sim/Obstacles") {
                try {
                    auto agent_array_msg = std::make_unique<social_msgs::msg::AgentStateArray>();
                    agent_array_msg->header.frame_id = std::to_string(frame_id_counter_++);

                    auto ros_message = std::make_shared<example_interfaces::msg::String>();
                    rclcpp::SerializedMessage rcl_serialized_message(*serialized_message_from_bag->serialized_data);
                    serialization.deserialize_message(&rcl_serialized_message, ros_message.get());

                    json outer_json = json::parse(ros_message->data);
                    if (outer_json.is_object()) {
                        // We iterate through each object in the per-frame message
                        for (auto it = outer_json.begin(); it != outer_json.end(); ++it) {
                            const std::string& key = it.key();

                            // Neet to modify here: now we are seeing all the agents at a given frame
                            // but for a robot FOV, it only sees limited number of agents at each frame,
                            // and the agents change constantly.
                            if (key.find("BP_CrownRandom_C_") == 0) {
                                try {
                                    // Find the position of the last underscore
                                    size_t last_underscore = key.find_last_of('_');
                                    if (last_underscore != std::string::npos) {
                                        // Get the substring after the last underscore
                                        std::string number_part = key.substr(last_underscore + 1);
                                        // Convert the numeric part to an integer
                                        int agent_id = std::stoi(number_part);

                                        json inner_json = json::parse(it.value().get<std::string>());

                                        social_msgs::msg::AgentState agent_state;
                                        agent_state.id = agent_id;
                                        agent_state.x = inner_json.value("x", 0.0);
                                        agent_state.y = inner_json.value("y", 0.0);
                                        agent_array_msg->agents.push_back(agent_state);
                                    }
                                } catch (const std::exception& e_inner) {
                                    // Handle errors from stoi or json::parse
                                    RCLCPP_WARN(this->get_logger(), "Could not process agent '%s'. Error: %s", key.c_str(), e_inner.what());
                                }
                            }
                        }
                    }
                    if (!agent_array_msg->agents.empty()) {
                        // We print message info published to /raw_agents every 100 frames
                        if (frame_id_counter_ % 100 == 0) {
                            RCLCPP_INFO(
                                this->get_logger(), 
                                "Frame %d: Processed %zu matching pedestrians (out of %zu total obstacles).",
                                frame_id_counter_,
                                agent_array_msg->agents.size(),
                                outer_json.size()
                            );
                        }
                        publisher_->publish(std::move(agent_array_msg));
                    }
                } catch (const std::exception& e_outer) {
                    RCLCPP_WARN(this->get_logger(), "Outer JSON parse error: %s", e_outer.what());
                }
            }
        }
        RCLCPP_INFO(this->get_logger(), "Bag processing complete. Shutting down.");
        rclcpp::shutdown();
    }
private:
    rclcpp::Publisher<social_msgs::msg::AgentStateArray>::SharedPtr publisher_;
    std::string bag_path_;
    int frame_id_counter_;
};

int main(int argc, char * argv[])
{
    if (argc < 2) {
        std::cerr << "Error: Missing bag file path argument." << std::endl;
        std::cerr << "Usage: " << argv[0] << "<path-to-bag-directory>" << std::endl;
        return 1;
    }
    std::string bag_path = argv[1];
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DataCleanerNode>(bag_path);
    std::thread processing_thread(
        [node]() {
            node->process_bag();
        }
    );
    rclcpp::spin(node);
    processing_thread.join();
    rclcpp::shutdown();
    return 0;
}