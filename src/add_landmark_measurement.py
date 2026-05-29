import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # X(4) was just added in 8.1 so it exists in initial_estimate but not in result
    # (result is from the optimization before X(4) was added).
    # L(2) comes from result, which has the best optimized landmark position.
    pose4 = initial_estimate.atPose2(X(4))
    landmark2 = result.atPoint2(L(2))

    # Compute the vector from X(4) to L(2) in the world frame
    dx = landmark2[0] - pose4.x()
    dy = landmark2[1] - pose4.y()

    # Range is the Euclidean distance
    distance = math.sqrt(dx**2 + dy**2)

    # Bearing is the angle to the landmark relative to the robot's heading
    bearing = gtsam.Rot2(math.atan2(dy, dx) - pose4.theta())

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), bearing, distance, MEASUREMENT_NOISE))
    return graph