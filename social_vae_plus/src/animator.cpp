#include "social_vae_plus/animator.hpp"
#include "social_vae_plus/dataloader.hpp"
#include "social_vae_plus/predictor.hpp"
#include <iostream>

class Predictor;
class DataLoader;

Animator::Animator(const std::string& window_title, unsigned int width, unsigned int height, sf::FloatRect world_bounds)
    : window_(sf::VideoMode(width, height), window_title) {
        // Set up the camera view
        // Add some padding around the edges for better visibility
        const float padding = 2.0f; 
        sf::FloatRect padded_bounds = world_bounds;
        padded_bounds.left -= padding;
        padded_bounds.top -= padding;
        padded_bounds.width += (2 * padding);
        padded_bounds.height += (2 * padding);


        //view_.setSize(40.0f, 30.0f); // Set a reasonable world size to view
        //view_.setCenter(10.0f, 10.0f); // Center the view on a point

        view_.reset(padded_bounds);
        window_.setView(view_);

        // Load a font for text
        // Make sure 'arial.ttf' is in your project directory or provide a full path
        /*
        if (!font_.loadFromFile("arial.ttf")) {
            std::cerr << "Error: Could not load font 'arial.ttf'. Make sure it's in the execution directory." << std::endl;
        }

        frame_text_.setFont(font_);
        frame_text_.setCharacterSize(24);
        frame_text_.setFillColor(sf::Color::Black);
        */
    }

void Animator::process_events() {
    sf::Event event;
    while (window_.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window_.close();
        }
        // Add a pause feature
        if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space) {
            is_paused_ = !is_paused_;
        }
    }
}

sf::Vector2f Animator::world_to_screen(sf::Vector2f world_pos) {
    // For SFML, a larger Y value means further down the screen.
    //return sf::Vector2f(world_pos.x, -world_pos.y);
    return world_pos;
}

void Animator::update(DataLoader& data_loader, Predictor& predictor) {
    auto all_frames = data_loader.get_unique_frame_ids();
    if (current_frame_idx_ >= all_frames.size()) {
        std::cout << "End of dataset reached. Exiting." << std::endl;
        is_paused_ = true; // Pause the animation if we reach the end
        return;
    }
    //std::cout << "Current frame idx: " << current_frame_idx_ << std::endl;

    int current_frame_id = all_frames[current_frame_idx_];
    //frame_text_.setString("Frame: " + std::to_string(current_frame_id));

    // Same batching logic as before
    const int OB_HORIZON = 8;
    std::vector<int> pids_for_prediction;
    std::vector<torch::Tensor> history_batch_list;

    auto persons_in_frame = data_loader.get_persons_in_frame(current_frame_id);

    // Mark all current visuals as potentially missing
    for (auto& pair : agent_visuals_) {
        pair.second.frames_since_seen++;
    }
    
    // Modify the code here!
    for (int person_id : persons_in_frame) {
        //std::cout << "Person ID: " << person_id << std::endl;
        // Get the history for this person
        torch::Tensor history = data_loader.get_history(person_id, current_frame_idx_, OB_HORIZON);
        //std::cout << "History for agent: " << person_id << " is: " << history << std::endl;

        // Check if the history is valid. If not, don't do anything with this person.
        /*
        if (history.numel() > 0) {
            pids_for_prediction.push_back(person_id);
            history_batch_list.push_back(history);
        }
        */
        if (history.numel() == 0) {
            continue;
        }
        pids_for_prediction.push_back(person_id);
        history_batch_list.push_back(history);
    

        // Update or create visuals for this agent
        if (agent_visuals_.find(person_id) == agent_visuals_.end()) {
            // New agent found, create visuals
            AgentVisuals visuals;
            // box size and origin
            visuals.box.setSize(sf::Vector2f(BOX_SIZE, BOX_SIZE));
            visuals.box.setOrigin(BOX_SIZE / 2, BOX_SIZE / 2);
            // box color
            sf::Color color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);
            visuals.box.setFillColor(color);
            // history and prediction trails
            visuals.history_trail.setPrimitiveType(sf::LinesStrip);
            visuals.prediction_trail.setPrimitiveType(sf::LinesStrip);
            // texts
            /*
            visuals.id_text.setFont(font_);
            visuals.id_text.setString(std::to_string(person_id));
            visuals.id_text.setCharacterSize(15);
            visuals.id_text.setFillColor(sf::Color::Black);
            */
            agent_visuals_[person_id] = visuals;
            
        }

        // Update position and trail for this agent
        //std::cout << "fuck one 1" << std::endl;
        AgentVisuals& visuals = agent_visuals_.at(person_id);
        //std::cout << "fuck one 2" << std::endl;
        visuals.frames_since_seen = 0; // Reset missing counter

        //torch::Tensor current_pos_tensor = history.slice(0, OB_HORIZON-1, OB_HORIZON).squeeze().slice(0, 0, 2);
        //sf::Vector2f world_pos(current_pos_tensor[0].item<float>(), current_pos_tensor[1].item<float>());
        torch::Tensor current_state_tensor = history.slice(0, OB_HORIZON - 1, OB_HORIZON).squeeze();
        float world_x = current_state_tensor[0].item<float>();
        float world_y = current_state_tensor[1].item<float>();

        //visuals.box.setPosition(world_to_screen(world_pos));
        //visuals.box.setPosition(world_pos);
        //visuals.id_text.setPosition(world_to_screen(world_pos) + sf::Vector2f(0.2f, -0.2f));
        sf::Vector2f world_pos(world_x, world_y);
        visuals.box.setPosition(world_to_screen(world_pos));

        //visuals.positions.push_back(world_to_screen(world_pos));
        //visuals.positions.push_back(world_pos);
        visuals.positions.push_back(world_to_screen(world_pos));
        if (visuals.positions.size() > 50) {
            // Keep trail length managebale
            visuals.positions.erase(visuals.positions.begin());
        }

        visuals.history_trail.clear();
        for (const auto& pos : visuals.positions) {
            visuals.history_trail.append(sf::Vertex(pos, visuals.box.getFillColor()));
        }

        visuals.prediction_trail.clear(); // Clear old predicitons
    }

    // Run prediction and update prediction trails
    if (!pids_for_prediction.empty()) {
        torch::Tensor history_batch = torch::stack(history_batch_list, 1);
        int n_agents = pids_for_prediction.size();

        // Simplified neighbor logic for visualization
        torch::Tensor neighbor_batch = torch::full({OB_HORIZON, n_agents, 1, 6}, 1e9);

        torch::Tensor prediction_batch = predictor.predict(history_batch, neighbor_batch);

        if (prediction_batch.numel() > 0) {
            for (int i = 0; i < n_agents; ++i) {
                int person_id_to_update = pids_for_prediction[i];
                if (agent_visuals_.find(person_id_to_update) == agent_visuals_.end()) {
                    std::cerr << "CRITICAL ERROR: Agent ID " << person_id_to_update
                            << " is in the prediction batch but NOT in the visuals map."
                            << " This should not happen. Skipping update for this agent." << std::endl;
                    continue; // Skip this agent to prevent crash
                }
                //std::cout << "fuck two 1" << std::endl;
                AgentVisuals& visuals = agent_visuals_.at(person_id_to_update);
                //std::cout << "fuck two 2" << std::endl;
                torch::Tensor single_pred = prediction_batch.select(1, i);

                // Add current position as the start of the prediction trail
                visuals.prediction_trail.append(sf::Vertex(visuals.box.getPosition(), sf::Color::Red));
                for (int j = 0; j < single_pred.size(0); ++j) {
                    float pred_x = single_pred[j][0].item<float>();
                    float pred_y = single_pred[j][1].item<float>();
                    //sf::Vector2f pred_pos(single_pred[j][0].item<float>(), single_pred[j][1].item<float>());
                    //visuals.prediction_trail.append(sf::Vertex(world_to_screen(pred_pos), sf::Color::Red));
                    sf::Vector2f pred_pos(pred_x, pred_y);
                    visuals.prediction_trail.append(sf::Vertex(world_to_screen(pred_pos), sf::Color::Red));
                }
            }
        }
    }
    cleanup_agents();
    current_frame_idx_++;
}

void Animator::cleanup_agents() {
    std::vector<int> ids_to_remove;
    for (auto const& [id, visuals] : agent_visuals_) {
        if (visuals.frames_since_seen > MISSING_THRESHOLD) {
            ids_to_remove.push_back(id);
        }
    }

    for (int id : ids_to_remove) {
        agent_visuals_.erase(id);
    }
}

void Animator::render() {
    window_.clear(sf::Color::White);

    // Draw everything
    for (auto const& [id, visuals] : agent_visuals_) {
        if (visuals.frames_since_seen == 0) {
            // Only draw visible agents
            window_.draw(visuals.history_trail);
            window_.draw(visuals.prediction_trail);
            window_.draw(visuals.box);
            //window_.draw(visuals.id_text);
        }
    }

    // Draw UI elements 
    /*
    window_.setView(window_.getDefaultView());
    window_.draw(frame_text_);
    window_.setView(view_);
    */

    window_.display();
}

void Animator::run(DataLoader& data_loader, Predictor& predictor) {
    std::cout << "Start running animator." << std::endl;
    sf::Clock clock;
    const sf::Time time_per_frame = sf::seconds(1.f / 10.f); // Control animation speed (10 FPS)

    while (window_.isOpen()) {
        //std::cout << "Enter the while loop." << std::endl;
        process_events();

        // clock.getElapsedTime() >= time_per_frame
        if (!is_paused_ && clock.getElapsedTime() >= time_per_frame) {
            //std::cout << "Ready to update the visualizer." << std::endl;
            update(data_loader, predictor);
            clock.restart();
        }

        render();
    }
}