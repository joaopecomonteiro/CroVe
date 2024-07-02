import numpy as np
# import cv2
import yaml
import pdb
import skimage as ski



def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2. T_x2_x1

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

        yaw, pitch, roll in degree

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1) # wP = x1_to_world * 1P, so x1_to_world is Tw1
    x2_to_world = x_to_world(x2) # Tw2
    world_to_x2 = np.linalg.inv(x2_to_world) # T2w

    transformation_matrix = np.dot(world_to_x2, x1_to_world) # T2w * Tw1 = T21
    return transformation_matrix



def render_polygon(mask, polygon, resolution, size_side, value=1):
    if len(polygon) == 0:
        return
    polygon = polygon / resolution
    polygon = np.round(polygon).astype(int)
    rows = polygon[:, 1]
    cols = polygon[:, 0]


    ix = (rows<size_side) & (rows>=0) & (cols<size_side) & (cols>=0)
    rows = rows[ix]
    cols = cols[ix]
    if len(rows)>0 and len(cols)>0:
        rr, cc = ski.draw.polygon(rows, cols)

        mask[:, rr, cc] = value


def get_car_masks(camera, config, yaml_file):
    cam_to_world = x_to_world(yaml_file[camera]["cords"])

    map_extents = config.map_extents
    map_resolution = config.map_resolution

    x1, y1, x2, y2 = map_extents
    mask_width = int((y2 - y1) / map_resolution)
    mask_height = int((x2 - x1) / map_resolution)
    masks = np.zeros((2, mask_height, mask_width), dtype=np.uint8)

    for vehicle in yaml_file["vehicles"].keys():

        roll, yaw, pitch = yaml_file["vehicles"][vehicle]["angle"]
        yaw_rad = np.deg2rad(yaw)

        extent, center, location = yaml_file["vehicles"][vehicle]["extent"], yaml_file["vehicles"][vehicle]["center"], yaml_file["vehicles"][vehicle]["location"]

        center_bbox = np.array(location) - np.array(center)
        
        corners = np.array([
            [-extent[0], -extent[1], -extent[2], 1],  # Back bottom left
            [-extent[0], extent[1], -extent[2], 1],   # Back bottom right
            [extent[0], extent[1], -extent[2], 1],    # Front bottom right
            [extent[0], -extent[1], -extent[2], 1],   # Front bottom left
            [-extent[0], -extent[1], extent[2], 1],   # Back top left
            [-extent[0], extent[1], extent[2], 1],    # Back top right
            [extent[0], extent[1], extent[2], 1],     # Front top right
            [extent[0], -extent[1], extent[2], 1]     # Front top left
        ])

        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0, 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rotated_corners = (rotation_matrix @ corners.T).T
        bbox_world = rotated_corners[:, :3] + center_bbox


        cam_to_world = x_to_world(yaml_file[camera]["cords"])
        world_to_cam = np.linalg.inv(cam_to_world)

        bbox_cam = (world_to_cam @ np.concatenate((bbox_world, np.ones((len(bbox_world), 1))), -1).T).T
        bbox_cam = bbox_cam[:4, :2]

        bbox_cam_tmp = bbox_cam.copy()
        bbox_cam[:, 0] = bbox_cam_tmp[:, 1]
        bbox_cam[:, 1] = bbox_cam_tmp[:, 0]
        bbox_cam[:, 0] += 25
        

        render_polygon(masks[0], bbox_cam, map_extents, map_resolution)

    return masks.astype("bool")


def get_car_bboxs(config, yaml_file):

    self_to_world = x_to_world(yaml_file["true_ego_pos"])
    world_to_self = np.linalg.inv(self_to_world)

    bboxs = []
    for vehicle in yaml_file["vehicles"].keys():

        roll, yaw, pitch = yaml_file["vehicles"][vehicle]["angle"]
        yaw_rad = np.deg2rad(yaw)

        extent, center, location = yaml_file["vehicles"][vehicle]["extent"], yaml_file["vehicles"][vehicle]["center"], yaml_file["vehicles"][vehicle]["location"]

        center_bbox = np.array(location) - np.array(center)
        
        corners = np.array([
            [-extent[0], -extent[1], -extent[2], 1],  # Back bottom left
            [-extent[0], extent[1], -extent[2], 1],   # Back bottom right
            [extent[0], extent[1], -extent[2], 1],    # Front bottom right
            [extent[0], -extent[1], -extent[2], 1],   # Front bottom left
            [-extent[0], -extent[1], extent[2], 1],   # Back top left
            [-extent[0], extent[1], extent[2], 1],    # Back top right
            [extent[0], extent[1], extent[2], 1],     # Front top right
            [extent[0], -extent[1], extent[2], 1]     # Front top left
        ])

        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0, 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        rotated_corners = (rotation_matrix @ corners.T).T
        bbox_world = rotated_corners[:, :3] + center_bbox
        

        bbox_self = (world_to_self @ np.concatenate((bbox_world, np.ones((len(bbox_world), 1))), -1).T).T
        if bbox_self[0, 0] >= -51 and bbox_self[0, 1] >= -51 and bbox_self[4, 0] <= 51 and bbox_self[4, 1] <= 51:
            bboxs.append(bbox_world)

    return bboxs





def get_visible_mask(instrinsics, image_width, config):
    extents, resolution = config.map_extents, config.map_resolution
    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, y1, x2, y2 = extents
    x, y = np.arange(y1, y2, resolution), np.arange(x1, x2, resolution)
    ucoords = x / y[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)


def get_camera_angle(x, y):
    return np.degrees(np.arccos((x**2-y**2)/(x**2+y**2)))






