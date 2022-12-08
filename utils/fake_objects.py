import numpy as np

def create_virtual_obj(object_name, camera_dist):
    # this is half a big cylinder with a half small cylinder
    if object_name == "shape1":
        rot_angle = np.abs(np.random.rand())
        height = np.abs(np.random.rand() * 20)
        if 0.2 < rot_angle < 0.7:
            dist = 0.3 * camera_dist
        else:
            dist = 0.8 * camera_dist
        if height > 15:
            dist = camera_dist
    # this is cake with one slice missing
    elif object_name == "shape2":
        rot_angle = np.abs(np.random.rand())
        height = np.abs(np.random.rand() * 20)
        if 5 <= height < 10:
            if rot_angle < 0.8:
                dist = 0.4 * camera_dist
            else:
                dist = camera_dist
        elif 10 <= height <= 15:
            if rot_angle < 0.8:
                dist = 0.6 * camera_dist
            else:
                dist = camera_dist
        else:
            dist = camera_dist
    # this is complex shape
    elif object_name == "shape3":
        rot_angle = np.abs(np.random.rand())
        height = np.abs(np.random.rand() * 20)
        if height < 2 or height > 18:
            dist = camera_dist
        elif 2 <= height < 4:
            dist = 0.5 * camera_dist
        elif 4 <= height < 9:
            dist = 0.7 * camera_dist
        elif 9 <= height < 12:
            dist = 0.6 * camera_dist
        elif 12 <= height < 16:
            dist = 0.4 * camera_dist
        else:
            dist = 0.55 * camera_dist
        if 0 < rot_angle < 0.1 or 0.3 < rot_angle < 0.4 or 0.6 < rot_angle < 0.7:
            dist = camera_dist
    # this is a normal cylinder
    else:
        rot_angle = np.abs(np.random.rand())
        height = np.abs(np.random.rand() * 20)
        dist = camera_dist / 2.
    return rot_angle, height, dist
