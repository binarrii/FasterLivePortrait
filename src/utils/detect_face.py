import cv2


def reduce_face_change(x_d_i_info: dict):
    x_d_i_info['pitch'][0] = x_d_i_info['pitch'][0] * .5
    x_d_i_info['yaw'][0] = x_d_i_info['yaw'][0] * .7
    x_d_i_info['roll'][0] = x_d_i_info['roll'][0] * .7
    x_d_i_info['scale'][0] = x_d_i_info['roll'][0] * .5

    return (x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])


def face_rotate_too_much(x_d_i_info: dict):
    if x_d_i_info['pitch'][0] > 20 or x_d_i_info['pitch'][0] < -20 or x_d_i_info['yaw'][0] > 20 or x_d_i_info['yaw'][0] < -20 or x_d_i_info['roll'][0] > 20 or x_d_i_info['roll'][0] < -20:
        return True
    else:
        return False

