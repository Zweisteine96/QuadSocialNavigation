# How to run the SocialVAE Workspace
- Step 1: you need to download the LibTorch ZIP archives from (here)[https://docs.pytorch.org/cppdocs/installing.html]. Be careful if you need a CPU-only libtorch or a GPU-enabled libtorch. In my case I used CPU-only version.
- Step 2: extract the libtorch to the same directory as `/src`, i.e., you will have folders of `/src` and `/libtorch` both in the `social_vae_cpp_ws`.
- Step 3: to test the intergration with MUJOCO-UE module, you will need to have the `my_bag` folder in your `/social_vae_cpp_ws/src/social_pipeline_node/data/` directory, which contains the ROS2 bag data. 
- Step 4: before the compilation, remember to source your system-level ROS2 by running the command `source /opt/ros/humbel/setup.bash`. Then in the workspace of `social_vae_cpp_ws`, you can run the command `colcon build --symlink-install --cmake-args -DCMAKE_PREFIX_PATH=$(pwd)/libtorch` to compile the package.
- Step 5: then run the command `source install/setup.sh` to source the workspace.

Now you can test the functionalities of various nodes implemented in this package.
