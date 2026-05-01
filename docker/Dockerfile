FROM ros:humble

WORKDIR /langnav

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Build ROS2 workspace
RUN cd langnav/ros2_ws && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build

# Source ROS2
RUN echo "source /langnav/langnav/ros2_ws/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]
