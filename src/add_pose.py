import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X
from helperfunctions import add_pose_from_global

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):

    # The robot rotates ~45 degrees, moves ~2 meters forward, then rotates ~45 degrees more.
    # Global target pose for X(4): x = 4 + sqrt(2), y = sqrt(2), theta = pi/2
    pose3 = initial_estimate.atPose2(X(3))
    pose4_global = gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2)

    # Use the helper to add the odometry factor and initial estimate from the global pose
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(3),
        new_key=X(4),
        prev_pose=pose3,
        new_pose_global=pose4_global,
        odom_noise=ODOMETRY_NOISE
    )

    return graph, initial_estimate