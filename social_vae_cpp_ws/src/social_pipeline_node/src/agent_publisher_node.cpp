// Copyright 2025 CHENG GUO
// Licensed under the Apache License, Version 2.0

#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
//#include "ament_index_cpp/packages.hpp"
#include "rclcpp/rclcpp.hpp"
#include "social_msgs/msg/agent_state.hpp"
#include "social_msgs/msg/agent_state_array.hpp"

using namespace std::chrono_literals;

// Struct to hold one line of data from the raw txt file
struct RawPoint
{
    int frame_id;
    double person_id;
    double x;
    double y;
};

// This node simulates a perception system by publishing agent data from an existing file
class AgentPublisherNode : public rclcpp::Node
{
public:
    AgentPublisherNode()
    : Node("agent_publisher_node"), current_frame_idx_(0)
    {
        // 1. Declare and get parameters
        this->declare_parameter<std::string>("data_file_path", "");
        std::string data_file = this->get_parameter("data_file_path").as_string();

        // If the path is not provided, try to find it in the package's shared directory
        if (data_file.empty()) {
            RCLCPP_INFO(this->get_logger(), "data_file_path not set, searching in package share...");
            std::string package_share_path = ament_index_cpp::get_package_share_directory("social_pipeline_node");
            data_file = package_share_path + "/data/eth/test/biwi_eth.txt";
            /*
            try {
                std::string package_share_path = ament_index_cpp::get_package_share_directory("social_pipeline_node");
                data_file = package_share_path + "/data/eth/test/biwi_eth.txt";
            } catch (const ament_index_cpp::PackageNotFoundError & e) {
                RCLCPP_ERROR(this->get_logger(), "Package 'social_pipeline_node' not found. Existing.");
                rclcpp::shutdown();
                return;
            }
            */
        }
        RCLCPP_INFO(this->get_logger(), "Loading data from: %s", data_file.c_str());

        // 2. Load and process the raw data
        load_data(data_file);
        if (unique_frame_ids_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No data loaded from file. Shutting down.");
            rclcpp::shutdown();
            return;
        }

        // 3. Create ROS2 components
        // Create a publisher that sends AgentStateArray messages on the "/raw_agents" topic
        publisher_ = this->create_publisher<social_msgs::msg::AgentStateArray>("/raw_agents", 10);

        // Create a timer that calls the 'timer_callback' function every 250ms (4Hz)
        timer_ = this->create_wall_timer(250ms, std::bind(&AgentPublisherNode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "Agent Publisher Node has started successfully.");
    }
private:
    // This function reads the dataset file and organizes the data by frame ID
    void load_data(const std::string & file_path)
    {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open data file: %s", file_path.c_str());
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            double frame_d, pid_d, x_d, y_d;
            ss >> frame_d >> pid_d >> x_d >> y_d;
            if (ss) {
                // Store each point in a map, keyed by its frame ID
                frame_data_[static_cast<int>(frame_d)].push_back(
                    {(int)frame_d, (double)pid_d, x_d, y_d}
                );
            }
        }

        // Create a sorted list of unique frame IDs to iterate through in order
        for (auto const & [frame_id, points] : frame_data_) {
            unique_frame_ids_.push_back(frame_id);
        }
        std::sort(unique_frame_ids_.begin(), unique_frame_ids_.end());
    }

    // This function is called by the timer at a regular interval
    void timer_callback()
    {
        // Stop if we have published all frames
        if (current_frame_idx_ >= unique_frame_ids_.size()) {
            RCLCPP_INFO(this->get_logger(), "End of dataset reached. Publisher has stopped.");
            timer_->cancel();
            return;
        }

        // Get the ID of the current frame to publish
        int current_frame_id = unique_frame_ids_[current_frame_idx_];

        // Create a new message to fill with data
        auto msg = social_msgs::msg::AgentStateArray();
        msg.header.stamp = this->get_clock()->now();
        // We store the frame ID in the header's frame_id field for the next node to use
        msg.header.frame_id = std::to_string(current_frame_id);

        // Add all agents from the current frame to the message
        for (const auto & point : frame_data_[current_frame_id]) {
            social_msgs::msg::AgentState agent;
            agent.id = point.person_id;
            agent.x = point.x;
            agent.y = point.y;
            // vx, vy, ax, ay will be handled later
            msg.agents.push_back(agent);
        }

        // Publish the message
        publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published data for frame %d", current_frame_id);

        // Move to the next frame for the next callback
        current_frame_idx_++;
    }

    // ROS2 components
    rclcpp::Publisher<social_msgs::msg::AgentStateArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Data storage
    std::map<int, std::vector<RawPoint>> frame_data_;
    std::vector<int> unique_frame_ids_;
    size_t current_frame_idx_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AgentPublisherNode>());
    rclcpp::shutdown();
    return 0;
}