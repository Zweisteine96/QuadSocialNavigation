#ifndef SOCIAL_VAE_ANIMATOR_HPP
#define SOCIAL_VAE_ANIMATOR_HPP

#include <SFML/Graphics.hpp>
#include <map>
#include <vector>
#include <string>

// A struct to hold all visual elemetns for one person
struct AgentVisuals {
    sf::RectangleShape box;
    //sf::Text id_text;
    sf::VertexArray history_trail;
    sf::VertexArray prediction_trail;
    std::vector<sf::Vector2f> positions; // Store history for the trail
    int frames_since_seen = 0;
};

class Animator {
public:
    Animator(const std::string& window_title, unsigned int width, unsigned int height, sf::FloatRect world_bounds);

    // Main animation loop function
    void run(class DataLoader& data_loader, class Predictor& predictor);

private:
    void process_events();
    void update(class DataLoader& data_loader, class Predictor& predictor);
    void render();
    void cleanup_agents(); // Remove agents that have been for too long

    // Concert world coordinatedes to screen coordinates for drawing
    sf::Vector2f world_to_screen(sf::Vector2f world_pos);

    // Member variables
    sf::RenderWindow window_;
    sf::View view_; // Camera view, allowing for panning and zooming
    //sf::Font font_;

    std::map<int, AgentVisuals> agent_visuals_; // Map person_id to their visuals
    //sf::Text frame_text_; // Text to display the current frame ID

    // Simulation state
    int current_frame_idx_ = 0;
    bool is_paused_ = false;

    // Constants for the simulation
    const float BOX_SIZE = 0.3f;
    const int MISSING_THRESHOLD = 5; // Frames to wait before removing an agent
};


#endif