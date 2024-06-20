import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Load the original mesh
mesh_path = '/home/krishan/work/2024/repos/diffrobot/diffrobot/pose_extraction/FoundationPose/demo_data/saucer/mesh/mesh.obj'

original_mesh = trimesh.load(mesh_path)

# Scale the mesh to 7% of its original size
scaled_mesh = original_mesh.copy()
scaling_factor = 0.07 # 7 for cup; 11 for teapot; 7 for all others
scaled_mesh.apply_scale(scaling_factor)

scaled_mesh_path = mesh_path.replace('.obj', '_scaled.obj')
scaled_mesh.export(scaled_mesh_path)

# Shift the scaled mesh to the right
shift_vector = np.array([original_mesh.extents[0] * 1.5, 0, 0])
scaled_mesh.apply_translation(shift_vector)

# Create a scene and add both meshes
scene = trimesh.Scene()
scene.add_geometry(original_mesh, node_name='Original Mesh')
scene.add_geometry(scaled_mesh, node_name='Scaled Mesh')

# Show the scene
scene.show()

# Save the scaled mesh to a new file
# save in the same directory as the original mesh

