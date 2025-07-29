// Copyright 2025 CHENG GUO
// Licensed under the Apache License, Version 2.0

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
//#include "ament_index_cpp/packages.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "social_msgs/msg/agent_state_array.hpp"
#include "social_msgs/msg/predicted_path_array.hpp"
#include "social_pipeline_node/predictor.hpp"
#include "torch/torch.h"

// Configuration constants
const int OB_HORIZON = 8;
const double DELTA_T = 0.4;

class StateProcessorNode : public rclcpp::Node
{
public:
    StateProcessorNode()
    : Node("state_processor_node")
    {
        // 1. Get the parameters
        this->declare_parameter<std::string>("model_path", "");
        std::string model_path = this->get_parameter("model_path").as_string();
        if (model_path.empty()) {
            std::string package_share_path = ament_index_cpp::get_package_share_directory("social_pipeline_node");
            model_path = package_share_path + "/data/model/eth/social_vae_traced.pt";
            /*
            try {
                std::string package_share_path = ament_index_cpp::get_package_share_directory("social_pipeline_node");
                model_path = package_share_path + "/data/model/eth/social_vae_traced.pt";
            } catch (const ament_index_cpp::PackageNotFoundError & e) {
                RCLCPP_ERROR(this->get_logger(), "Package 'social_pipeline_node' not found. Exiting.");
                rclcpp::shutdown();
                return;
            }
            */
        }

        // 2. Initialize the Predictor model
        predictor_ = std::make_unique<Predictor>(model_path);

        // 3. Create ROS2 components
        // Subscribe to the raw agent data
        subscription_ = this->create_subscription<social_msgs::msg::AgentStateArray>(
            "/raw_agents", 10, std::bind(&StateProcessorNode::agent_callback, this, std::placeholders::_1)
        );

        // Publish the full kinematics state (pos, vel, acc) of agents with valid history
        state_publisher_ = this->create_publisher<social_msgs::msg::AgentStateArray>("/full_agent_states", 10);

        // Publish the predicted future paths for those agents
        path_publisher_ = this->create_publisher<social_msgs::msg::PredictedPathArray>("/predicted_paths", 10);

        RCLCPP_INFO(this->get_logger(), "State Processor Node has started successfully.");
    }
private:
    // Main callback function, triggered whenever new raw agent data arrives
    void agent_callback(const social_msgs::msg::AgentStateArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received data for frame %s", msg->header.frame_id.c_str());

        // 1. Ingest new data and update internal history buffer
        int current_frame_id = std::stoi(msg->header.frame_id);
        for (const auto & agent : msg->agents){
            // Create a 6-element tensor for {x, y, vx, vy, ax, ay}, initialized to NaN
            torch::Tensor features = torch::full({6}, std::numeric_limits<double>::quiet_NaN());
            features[0] = agent.x;
            features[1] = agent.y;
            // Store this new data point in our history map
            trajectories_[agent.id][current_frame_id] = features;
        }

        // 2. Calculate derivatives for all agents seen in this frame
        calculate_derivatives_for_frame(current_frame_id);

        // 3. Identify agents with enough history to be processed
        std::vector<int> pids_for_processing;
        std::vector<torch::Tensor> history_batch_list;
        auto full_states_msg = social_msgs::msg::AgentStateArray();
        full_states_msg.header = msg->header;

        for (const auto & agent : msg->agents) {
            // get_history will return a [OB_HORIZON, 6] tensor if history is complete and valid
            torch::Tensor history = get_history(agent.id, current_frame_id);
            if (history.numel() > 0) {
                pids_for_processing.push_back(agent.id);
                history_batch_list.push_back(history);

                // Prepare the message with the full current state of this agent
                torch::Tensor current_state = history.slice(0, OB_HORIZON-1, OB_HORIZON).squeeze();
                social_msgs::msg::AgentState full_state;
                full_state.id = agent.id;
                full_state.x = current_state[0].item<double>();
                full_state.y = current_state[1].item<double>();
                full_state.vx = current_state[2].item<double>();
                full_state.vy = current_state[3].item<double>();
                full_state.ax = current_state[4].item<double>();
                full_state.ay = current_state[5].item<double>();
                full_states_msg.agents.push_back(full_state);
            }
        }
        // Publish the full states of all valid agents
        state_publisher_->publish(full_states_msg);

        // 4. Run prediction model if there are nay valid agents
        if (pids_for_processing.empty()) {
            return; // Nothing to predict, done for this frame
        }

        RCLCPP_INFO(this->get_logger(), "Predicting for %zu agents.", pids_for_processing.size());

        // Prepare input batch of shape [seq_len, batch_size, feature_dims] for the model
        torch::Tensor history_batch = torch::stack(history_batch_list, 1);

        // NOTE: Use a placeholder for neighbors. We will come back for this later!
        torch::Tensor neighbors_batch = torch::full({OB_HORIZON, history_batch.size(1), 1, 6}, 1e9, torch::kFloat64);

        // Run the inference
        torch::Tensor prediction_batch = predictor_->predict(history_batch, neighbors_batch);

        // 5. Publish the predicted trajectories
        if (prediction_batch.numel() > 0) {
            auto paths_msg = social_msgs::msg::PredictedPathArray();
            paths_msg.header = msg->header;
            for (long i = 0; i < pids_for_processing.size(); ++i) {
                social_msgs::msg::PredictedPath pred_path;
                pred_path.id = pids_for_processing[i];
                pred_path.path.header = msg->header;
                torch::Tensor single_pred = prediction_batch.select(1, i); // Get prediction for single agent
                for (long j = 0; j < single_pred.size(0); ++j) {
                    geometry_msgs::msg::PoseStamped pose;
                    pose.header = msg->header;
                    pose.pose.position.x = single_pred[j][0].item<double>();
                    pose.pose.position.y = single_pred[j][1].item<double>();
                    pred_path.path.poses.push_back(pose);
                }
                paths_msg.predictions.push_back(pred_path); // Store predicted paths for all agents
            }
            path_publisher_->publish(paths_msg);
        }
    }

    // Calculate derivatves for all agents present in a given frame
    void calculate_derivatives_for_frame(int frame_id)
    {
        // A simplified version of the manual derivative calculation.
        // We assume that if an agent is present at frame_id, it was also present
        // somewhere else in the previous frames it was seen
        for (auto & person_pair : trajectories_) {
            auto & trajectory = person_pair.second;
            // Only calculate for this agent if it is present in the current frame
            if (trajectory.count(frame_id)) {
                if (trajectory.size() < 2) {continue;}
                auto it = trajectory.rbegin(); // Iterator to the most recent entry
                int frame2_id = it->first;
                torch::Tensor & features2  =it->second;
                torch::Tensor pos2  =features2.slice(0, 0, 2);

                it++; // Move to the previous entry
                int frame1_id = it->first;
                torch::Tensor pos1 = it->second.slice(0, 0, 2);

                double dt = (frame2_id - frame1_id) * DELTA_T;
                if (dt > 1e-6) {
                    torch::Tensor vel = (pos2 - pos1) / dt;
                    features2.slice(0, 2, 4) = vel;
                    if (trajectories_[person_pair.first].count(frame1_id) && 
                    !torch::any(torch::isnan(trajectories_[person_pair.first].at(frame1_id).slice(0, 2, 4))).item<bool>()) {
                        torch::Tensor vel1 = trajectories_[person_pair.first].at(frame1_id).slice(0, 2, 4);
                        torch::Tensor acc = (vel - vel1) / dt;
                        features2.slice(0, 4, 6) = acc;
                    } else {
                        features2.slice(0, 4, 6) = torch::zeros({2}, torch::kFloat64);
                    }
                }
            }
        }
    }

    // Retrieve a valid history tensor for a given agent at a frame
    torch::Tensor get_history(int person_id, int current_frame_id)
    {
        if (trajectories_.find(person_id) == trajectories_.end()) {
            return torch::Tensor();
        }
        const auto & trajectory = trajectories_.at(person_id);

        // FInd the index of the current_frame_id in a sorted list of this agent's frames
        std::vector<int> frames;
        for (const auto & pair : trajectory) {
            frames.push_back(pair.first);
        }
        std::sort(frames.begin(), frames.end());

        auto it = std::find(frames.begin(), frames.end(), current_frame_id);
        if (it == frames.end()) return torch::Tensor(); // Frame not found for this agent

        // Check if there are enough historical frames before this frame
        int current_idx = std::distance(frames.begin(), it);
        if (current_idx < OB_HORIZON-1) {
            return torch::Tensor(); // Not enough history length
        }

        std::vector<torch::Tensor> history_features;
        for (int i = 0; i < OB_HORIZON; ++i) {
            int frame_to_get = frames[current_idx - (OB_HORIZON-1-i)];
            torch::Tensor feats = trajectory.at(frame_to_get);
            // Ensure derivatives have been calcualted (not NaN)
            if (torch::any(torch::isnan(feats)).item<bool>()) {
                return torch::Tensor();
            }
            history_features.push_back(feats);
        }
        return torch::stack(history_features).to(torch::kFloat64);
    }

    // ROS2 components and state
    std::unique_ptr<Predictor> predictor_;
    rclcpp::Subscription<social_msgs::msg::AgentStateArray>::SharedPtr subscription_;
    rclcpp::Publisher<social_msgs::msg::AgentStateArray>::SharedPtr state_publisher_;
    rclcpp::Publisher<social_msgs::msg::PredictedPathArray>::SharedPtr path_publisher_;

    // Internal state: map<person_id, map<frame_id, tensor[6]>>
    std::map<int, std::map<int, torch::Tensor>> trajectories_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StateProcessorNode>());
    rclcpp::shutdown();
    return 0;
}