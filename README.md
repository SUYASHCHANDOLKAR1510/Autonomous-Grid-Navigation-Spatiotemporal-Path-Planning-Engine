#### Autonomous Grid Navigation & Spatiotemporal Path Planning Engine

A real-time autonomous navigation, localization, and motion-control stack engineered in Python for a differential-drive mobile robot operating within a high-frequency **MuJoCo** physics simulation. Built on an event-driven **MQTT** pub/sub telemetry bus, the controller executes 100 Hz closed-loop state estimation, spatiotemporal A* trajectory optimization, and predictive obstacle evasion under strict latency constraints. 

#### System Architecture


ASCII

┌─────────────────────┐  100 Hz Sensor Frames   ┌──────────────────────┐
│ MuJoCo Physics Env  │ ──────────────────────▶ │  Mosquitto MQTT Bus  │
│ (500 Hz Dynamics)   │ ◀────────────────────── │   (Pub/Sub Broker)   │
└─────────────────────┘  Wheel Velocity (rad/s) └──────────┬───────────┘
                                                           │
                             ┌─────────────────────────────┴─────────────────────────────┐
                             ▼                                                           ▼
             ┌───────────────────────────────┐                           ┌───────────────────────────────┐
             │   State Estimation & Fusion   │                           │   Spatiotemporal A* Planner   │
             │ • Wheel Encoder Dead-Reckoning│                           │ • Multi-Target Route Search   │
             │ • Bias-Calibrated Yaw Gyro    │ ──▶ Current Pose [x,y,θ] ─▶ • Forward Obstacle Prediction │
             │ • 4-Ray ToF Wall Correction   │                           │ • Dynamic Evasion & Replanning│
             └───────────────────────────────┘                           └───────────────┬───────────────┘
                                                                                         │
                                                                                         ▼
                                                                         ┌───────────────────────────────┐
                                                                         │   Motion Primitive Executor   │
                                                                         │ • Trapezoidal Velocity Profile│
                                                                         │ • Circular Arc Cornering      │
                                                                         │ • Lateral Wall-Centering PID  │
                                                                         └───────────────────────────────┘

#### Core Engineering Highlights

* **Spatiotemporal A* Trajectory Optimization:** Formulates multi-target collection and exit routing as a time-expanded graph search over (row, col, heading, target_bitmask, predicted_obstacle_states), weighting kinematic transition costs (straight runs, 90° arcs, and 180° pivots) alongside forward-simulated obstacle trajectories.
* **Sensor Fusion & Odometry Correction:** Combines cumulative wheel-encoder dead-reckoning and online bias-calibrated Z-axis gyroscope integration with a 4-ray axis-aligned Time-of-Flight (ToF) rangefinder model to suppress drift and snap longitudinal/lateral pose against known map geometry.
* **Kinodynamic Motion Primitives:** Decomposes discrete directional paths into continuous motion primitives—featuring trapezoidal velocity profiling, continuous circular-arc cornering without full stops, and closed-loop lateral wall-centering control.
* **Predictive Threat Monitoring & Evasion:** Continuously evaluates pursuit horizons against live dynamic obstacle telemetry; triggers sub-second emergency braking, cell-centering alignment, and minimax-style survival replanning when collision margins drop below threshold.
* **100 Hz Event-Driven Control Loop:** Uses paho-mqtt over a local Mosquitto broker to decouple physics simulation from control computation, streaming clamped differential-drive angular velocity commands (rad/s) on every sensor tick.

#### Tech Stack

* **Core Language:** Python 3.10+
* **Physics & Simulation:** MuJoCo
* **Middleware / IPC:** MQTT (paho-mqtt, Eclipse Mosquitto)
* **Algorithms & Control:** Spatiotemporal A*, BFS Distance Fields, Sensor Fusion, Differential-Drive Kinematics, PID Control

