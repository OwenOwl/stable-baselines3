import pickle
import open3d as o3d
import cv2
import os
import numpy as np

def numpy_to_point_cloud(data):
    """Convert Numpy array to Open3D point cloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data)
    return pc

def set_camera_pose(vis, camera_pose):
    """Set camera pose."""
    ctr = vis.get_view_control()
    ctr.set_lookat(camera_pose["lookat"])
    ctr.set_up(camera_pose["up"])
    ctr.set_front(camera_pose["front"])
    ctr.set_zoom(camera_pose["zoom"])


def capture_frame(vis):
    """Capture frame for the current view."""
    image = vis.capture_screen_float_buffer(False)
    return (np.asarray(image) * 255).astype(np.uint8)

def create_video(frames, output_path='output.mp4', fps=30):
    """Create a video from a list of frames."""
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for frame in frames:
        out.write(frame)

    out.release()

# Load your point cloud data here (this is just an example)

traj_root = "real_trajs_0129"
# traj_root = "sim_trajs_0129"
traj_idx = 1

with open(f"{traj_root}/{traj_idx}/obs_traj.pkl", "rb") as f:
    traj = pickle.load(f)

print(traj[0])
point_clouds = [
    obs["relocate-point_cloud"] for obs in traj
]  # Replace with your sequence of point clouds


# camera_pose = {
#     "lookat": [0, 0, 0],  # Point the camera is looking at
#     "up": [0, 0, 1],      # Up direction of the camera
#     "front": [0, 1, 0],  # Front direction of the camera
#     "zoom": 0.01           # Zoom level
# }

# frames = []
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# for data in point_clouds:
#     pc = numpy_to_point_cloud(data)
#     vis.clear_geometries()
#     vis.add_geometry(pc)
#     set_camera_pose(vis, camera_pose)
#     vis.poll_events()
#     vis.update_renderer()
#     frame = capture_frame(vis)
#     frames.append(frame)

# vis.destroy_window()

# # Create the video
# create_video(frames, "real_pc_001.mp4")


new_points_for_visual = point_clouds[10][:,:3]
print(new_points_for_visual.shape)
camera_obs = o3d.geometry.PointCloud()
camera_obs.points = o3d.utility.Vector3dVector(new_points_for_visual)
camera_obs.paint_uniform_color([0, 0, 1])

# img_pc_for_visual = img_pc[:,:3]
# img = o3d.geometry.PointCloud()
# img.points = o3d.utility.Vector3dVector(img_pc_for_visual)
# img.paint_uniform_color([1, 0, 0])
coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

o3d.visualization.draw_geometries([camera_obs, coordinate])