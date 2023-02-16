## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# First import the library
import pyrealsense2 as rs
import pclpy
from pclpy import pcl
import numpy as np

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
config = rs.config()
# Enable depth stream
config.enable_stream(rs.stream.depth)

# Start streaming with chosen configuration
pipe.start(config)

# We'll use the colorizer to generate texture for our PLY
# (alternatively, texture can be obtained from color or infrared stream)
colorizer = rs.colorizer()

try:
    # Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames()
    colorized = colorizer.process(frames)

    # # Create save_to_ply object
    # ply = rs.save_to_ply("1.pcd")
    #
    # # Set options to the desired values
    # # # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    # # ply.set_option(rs.save_to_ply.option_ply_binary, False)
    # # ply.set_option(rs.save_to_ply.option_ply_normals, True)
    #
    # print("Saving to 1.ply...")
    # # Apply the processing block to the frameset which contains the depth frame and the texture
    # ply.process(colorized)
    # print("Done")

    depth_intrinsics = rs.video_stream_profile(colorized.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    out = np.empty((h, w, 3), dtype=np.uint8)
    out.fill(0)
    points = pc.calculate(colorized)

    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    pc2 = pcl.PointCloud.PointXYZ(verts)
    writer = pcl.io.PCDWriter()
    writer.writeBinary("./PointCloud.pcd", pc2)
finally:
    pipe.stop()