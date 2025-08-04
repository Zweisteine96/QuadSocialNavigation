// Copyright 2025 CHENG GUO
// Licensed under the Apache License, Version 2.0

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <deque>
#include <chrono>
#include <limits>
#include <vector>
#include <set>

#include "rclcpp/rclcpp.hpp"
#include "social_msgs/msg/agent_state_array.hpp"
#include "social_msgs/msg/predicted_path_array.hpp"
#include "SFML/Graphics.hpp"

using namespace std::chrono_literals;

// A struct to hold all visual elements for one agent
struct AgentVisual
{
    sf::RectangleShape box;
    sf::VertexArray history_trail;
    std::deque<sf::Vector2f> positions;
};

// This node visualizes the output of the pipeline
class VisualizerNode : public rclcpp::Node
{
public:
    VisualizerNode()
    : Node("visualizer_node")
    {
        // 1. Create subscribers
        state_sub_ = this->create_subscription<social_msgs::msg::AgentStateArray>(
            "/full_agent_states", 10,
            std::bind(&VisualizerNode::state_callback, this, std::placeholders::_1)
        );

        path_sub_ = this->create_subscription<social_msgs::msg::PredictedPathArray>(
            "/predicted_paths", 10,
            std::bind(&VisualizerNode::path_callback, this, std::placeholders::_1)
        );

        srand(time(NULL));

        // 2. Launch the display loop in a separate thread
        // The main thread will be busy with rclcpp::spin(), so all SFML
        // windowing and drawing must happen in a different thread.
        display_thread_ = std::thread(&VisualizerNode::display_loop, this);
        RCLCPP_INFO(this->get_logger(), "Visualizer Node has started.");
    }
private:
    // Callback for receiving the full state of all current agents
    void state_callback(const social_msgs::msg::AgentStateArray::SharedPtr msg)
    {
        // A mutex lock ensures that we don't write to this data while
        // the display thread is trying to read it.
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_states_ = *msg;
    }

    // Callback for receiving all predicted paths
    void path_callback(const social_msgs::msg::PredictedPathArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_paths_ = *msg;
    }

    // The main SFML drawing loop
    void display_loop()
    {
        sf::RenderWindow window(sf::VideoMode(1280, 1024), "Social Pipeline Visualizer");
        // This view setup should be adjusted based on dataset bounds for a perfect fit.
        //sf::View view(sf::FloatRect(0.f, 0.f, 20.f, 15.f));
        sf::View view = window.getDefaultView();
        //view.setCenter(10.f, 8.f);
        //window.setView(view);

        // Map to store the visual elements for each agent ID.
        std::map<int, AgentVisual> visuals;
        const float BOX_SIZE = 1.0f;

        // Select a subset of agents to animate
        const size_t MAX_AGENTS_TO_DISPLAY = 30;
        std::set<int> selected_agent_ids;

        while (window.isOpen() && rclcpp::ok()) {
            // Process window events (like closing the window).
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                    rclcpp::shutdown();
                }
            }

            // --- Safely copy data from ROS threads to this display thread ---
            social_msgs::msg::AgentStateArray states_to_draw;
            social_msgs::msg::PredictedPathArray paths_to_draw;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                states_to_draw = latest_states_;
                // Only draw predictions that correspond to the current state frame.
                // if (latest_states_.header.stamp == latest_paths_.header.stamp) {
                //     paths_to_draw = latest_paths_;
                // }
                if (!latest_states_.agents.empty()) {
                    paths_to_draw = latest_paths_;
                }
            }

            // Fix bug: auto tracking camera
            if (!states_to_draw.agents.empty()) {
                float min_x = std::numeric_limits<float>::max();
                float max_x = std::numeric_limits<float>::lowest();
                float min_y = std::numeric_limits<float>::max();
                float max_y = std::numeric_limits<float>::lowest();

                for (const auto& agent : states_to_draw.agents) {
                    if (selected_agent_ids.count(agent.id)) {
                        min_x = std::min(min_x, (float)agent.x); max_x = std::max(max_x, (float)agent.x);
                        min_y = std::min(min_y, (float)-agent.y); max_y = std::max(max_y, (float)-agent.y);
                    }
                    // min_x = std::min(min_x, (float)agent.x);
                    // max_x = std::max(max_x, (float)agent.x);
                    // min_y = std::min(min_y, (float)-agent.y);
                    // max_y = std::max(max_y, (float)-agent.y);
                }

                // If we haven't found any of our selected agents yet, don't update the camera
                if (min_x != std::numeric_limits<float>::max()) {
                    sf::Vector2f center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
                    sf::Vector2f size(max_x - min_x, max_y - min_y);

                    float view_size = std::max(size.x, size.y) * 1.2f;
                    view_size = std::max(view_size, 10.0f);
                    view_size = std::min(view_size, 25.0f); 

                    view.setCenter(center);
                    view.setSize(view_size, view_size * (1024.0f / 1280.0f));
                    window.setView(view);
                }

                // sf::Vector2f center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
                // sf::Vector2f size(max_x - min_x, max_y - min_y);

                // float view_size = std::max(size.x, size.y) * 1.5f;
                // view_size = std::max(view_size, 10.0f);

                // view.setCenter(center);
                // view.setSize(view_size, view_size * (1024.0f / 1280.0f));
                // window.setView(view);
            }
            // --- Drawing ---
            window.clear(sf::Color(50, 50, 50));

            // 1. Draw historical trails and current positions
            for (const auto & agent : states_to_draw.agents) {
                // If this is a new agent, decide if we should track it
                if (selected_agent_ids.find(agent.id) == selected_agent_ids.end()) {
                    if (selected_agent_ids.size() < MAX_AGENTS_TO_DISPLAY) {
                        selected_agent_ids.insert(agent.id);
                    } else {
                        continue; // Skip this agent, we have enough
                    }
                }

                // If this agent is not in our selected list, skip drawing it
                if (selected_agent_ids.count(agent.id) == 0) {
                    continue;
                }

                if (visuals.find(agent.id) == visuals.end()) {
                    visuals[agent.id].box.setSize({BOX_SIZE, BOX_SIZE});
                    visuals[agent.id].box.setOrigin(BOX_SIZE / 2, BOX_SIZE / 2);
                    visuals[agent.id].box.setFillColor(
                        sf::Color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55));

                    visuals[agent.id].box.setOutlineColor(sf::Color::Black);
                    visuals[agent.id].box.setOutlineThickness(0.05f);
                    visuals[agent.id].history_trail.setPrimitiveType(sf::LineStrip);
                }

                auto & v = visuals.at(agent.id);
                // Flip Y-axis for correct graphical display
                sf::Vector2f current_pos(agent.x, -agent.y);
                v.box.setPosition(current_pos);

                // Update the history trail
                v.positions.push_back(current_pos);
                if (v.positions.size() > 50) {v.positions.pop_front();}
                v.history_trail.clear();
                for (const auto & p : v.positions) {
                    v.history_trail.append({p, v.box.getFillColor()});
                }

                window.draw(v.history_trail);
                window.draw(v.box);
            }

            // 2. Draw predicted paths
            for (const auto & pred : paths_to_draw.predictions) {
                // sf::VertexArray pred_trail(sf::LineStrip);
                // for (const auto & pose : pred.path.poses) {
                //     pred_trail.append(
                //         {{(float)pose.pose.position.x, (float)-pose.pose.position.y}, sf::Color::Red});
                // }
                // window.draw(pred_trail);
                // Only draw predictions for agents we are tracking
                if (selected_agent_ids.count(pred.id)) {
                    sf::VertexArray pred_trail(sf::LineStrip);
                    for (const auto & pose : pred.path.poses) {
                        pred_trail.append({{(float)pose.pose.position.x, (float)-pose.pose.position.y}, sf::Color::Red});
                    }
                    window.draw(pred_trail);

                    // Extrapolate the final predicted path
                    if (pred.path.poses.size() >= 2) {
                        // Get the last two points of the prediction
                        const auto& last_pose = pred.path.poses.back();
                        const auto& second_last_pose = pred.path.poses[pred.path.poses.size() - 2];
                        
                        sf::Vector2f last_point((float)last_pose.pose.position.x, (float)-last_pose.pose.position.y);
                        sf::Vector2f second_last_point((float)second_last_pose.pose.position.x, (float)-second_last_pose.pose.position.y);

                        // Calculate the final velocity vector
                        sf::Vector2f final_velocity = last_point - second_last_point;

                        // Create a new point by extending from the last point along the velocity vector
                        float extrapolation_factor = 5.0f; // Make this line 5x longer
                        sf::Vector2f extrapolated_point = last_point + final_velocity * extrapolation_factor;

                        // Draw a "ghost" line from the last real point to the extrapolated point
                        sf::VertexArray ghost_trail(sf::Lines);
                        ghost_trail.append({last_point, sf::Color(255, 100, 100, 150)}); // Faded red
                        ghost_trail.append({extrapolated_point, sf::Color(255, 100, 100, 0)}); // Fade to transparent
                        window.draw(ghost_trail);
                    }
                }
            }
            window.display();
            std::this_thread::sleep_for(30ms);
        }
    }
    rclcpp::Subscription<social_msgs::msg::AgentStateArray>::SharedPtr state_sub_;
    rclcpp::Subscription<social_msgs::msg::PredictedPathArray>::SharedPtr path_sub_;

    // Threading and data synchronization
    std::thread display_thread_;
    std::mutex data_mutex_;

    // Shared data buffers
    social_msgs::msg::AgentStateArray latest_states_;
    social_msgs::msg::PredictedPathArray latest_paths_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualizerNode>());
    rclcpp::shutdown();
    return 0;
}