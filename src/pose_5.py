import math
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nOptimization result:\n{}".format(result))
    return result

def _pose_too_close_to_existing(pose_5, initial_estimate, threshold=0.5):
    for i in [1, 2, 3, 4]:
        try:
            p = initial_estimate.atPose2(X(i))
            dist = np.sqrt((p.x() - pose_5.x())**2 + (p.y() - pose_5.y())**2)
            if dist < threshold:
                return True
        except Exception:
            pass
    return False

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float("inf")

    for pose_key, pose_5 in pose_options.items():
        if _pose_too_close_to_existing(pose_5, initial_estimate):
            print(f"Pose {pose_key}: skipped (too close to existing pose)")
            continue

        for landmark in [1, 2]:
            g = gtsam.NonlinearFactorGraph(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            marginals = gtsam.Marginals(g, result)
            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum() +
                marginals.marginalCovariance(L(2)).sum()
            )

            print(f"Pose {pose_key}, Landmark {landmark}: sum of marginals = {sum_of_marginals:.4f}")

            if sum_of_marginals < best_sum:
                best_sum = sum_of_marginals
                best_pose = pose_key
                best_landmark = landmark

    print(f"\nBest pose: {best_pose}, Best landmark: {best_landmark}, Sum: {best_sum:.4f}")
    return best_pose, best_landmark, best_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = graph.clone()
            ie = gtsam.Values(initial_estimate)
            g, ie = add_pose(g, ie, pose_5)
            result = optimize(g, ie)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, ie)

            mse_X1 = math.sqrt((result.atPose2(X(1)).x()-0)**2 + (result.atPose2(X(1)).y()-0)**2 + (result.atPose2(X(1)).theta()-0)**2)
            mse_X2 = math.sqrt((result.atPose2(X(2)).x()-2)**2 + (result.atPose2(X(2)).y()-0)**2 + (result.atPose2(X(2)).theta()-0)**2)
            mse_X3 = math.sqrt((result.atPose2(X(3)).x()-4)**2 + (result.atPose2(X(3)).y()-0)**2 + (result.atPose2(X(3)).theta()-0)**2)

            sum_of_errors = mse_X1 + mse_X2 + mse_X3
            print(f"pose={pose_key}, landmark={landmark}, sum={sum_of_errors:.6e}")

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum