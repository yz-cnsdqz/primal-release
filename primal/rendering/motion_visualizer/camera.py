import numpy as np


def calculate_camera_position_from_trajectory(trajectory, fov=0.7, pitch=20, body_length=2):
    """
        Given the whole trajectory and camera foc and expected pitch angle, 
        calculate the camera position and look-at position, so that the entire 3D human motion can be covered in the camera view. 

    """
    # compensate for the body center to floor offset
    trajectory = trajectory - np.array([[0, 0.3, 0]])
    # Calculate the centroid (lookat point)
    centroid = np.mean(trajectory, axis=0)

    # Calculate the distances from the centroid to all trajectory points
    distances = np.linalg.norm(trajectory - centroid, axis=1)
    
    # Maximum distance to cover all points within the field of view
    # Instead of just covering the body center point, adding body_length to cover the whole body. 
    max_distance = max(distances)+body_length
    
    # Given the field of view and the max_distance, calculate the required distance of the camera from the centroid
    camera_distance = max_distance / np.tan(fov)
    
    # Calculate camera position: 
    # the key is to cover the depth-wise movement in the camera view, so it 
    camera_position = centroid + camera_distance * np.array([0, np.sin(np.deg2rad(pitch)), np.cos(np.deg2rad(pitch))])
    # camera_position = centroid + camera_distance * np.array([np.sin(np.deg2rad(pitch)), np.cos(np.deg2rad(pitch)), 0])
    
    return camera_position, centroid