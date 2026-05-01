#!/bin/bash
# Starts virtual display → VNC → noVNC websocket → ROS2 environment

set -e

# ── Virtual display ────────────────────────────────────────────────────────
Xvfb :1 -screen 0 1280x900x24 &
sleep 1

# ── VNC server (no password) ──────────────────────────────────────────────
x11vnc -display :1 -nopw -listen 0.0.0.0 -forever -shared -rfbport 5900 &
sleep 1

# ── noVNC websocket proxy → browser at :8080 ──────────────────────────────
websockify --web /usr/share/novnc 8080 localhost:5900 &
sleep 1

echo ""
echo "=========================================="
echo "  noVNC ready → open in browser:"
echo "  http://localhost:8080"
echo "=========================================="
echo ""

# ── Source ROS2 + workspace ────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
WS=/langnav/langnav/ros2_ws
if [ -f "$WS/install/setup.bash" ]; then
    source "$WS/install/setup.bash"
fi

export ROS_DOMAIN_ID=0

# ── Run command (or interactive bash) ─────────────────────────────────────
exec "$@"
