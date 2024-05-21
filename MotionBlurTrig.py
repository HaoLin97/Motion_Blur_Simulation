import math


def calc_camera_afov(sensor_height, focal_length):

    afov = 2 * math.degrees(math.atan(sensor_height/(2*focal_length)))

    return afov


if __name__ == '__main__':
    # Current values and settings simulate the sound lab setup
    # Inputs: FOV of camera, resolution of camera, distance between camera and target
    # standard car 4.9x1.8
    exposure_time = 15000  # in us
    target_x_real = 4.9  # in m
    target_y_real = 18  # in m
    target_speed = 27.778  # in ms-1   - moving 1 degree is equal to 4.01 cms-1
    target_direction = 'Horizontal'  # TODO Currently coded for horizontal only
    distance = 50  # distance between camera and target in m
    # Calculated
    camera_horizontal_afov = 99.32  # in degree
    camera_vertical_afov = 63.67  # in degree

    # Given by specs
    # camera_horizontal_afov = 94.8  # in degree
    # camera_vertical_afov = 77.9  # in degree

    camera_resolution_x = 4096
    camera_resolution_y = 2160
    focal_length = 6
    aperature = 1.8
    # target_x_image = 0
    # target_y_image = 0

    # Converting exposure in us to s
    exposure_time_seconds = exposure_time / 1000000

    # Distance moved by the target during the exposure time
    distance_moved = target_speed * exposure_time_seconds

    # The amount that the target has moved in percentage(0-1)
    percentage_moved = distance_moved/target_x_real

    # The amount of pixels that represents each degree
    px_per_deg = camera_resolution_x/camera_horizontal_afov

    # The angle of view for the target, from the camera pov.
    target_aov = 2 * math.degrees(math.atan((target_x_real/2)/distance))

    # The amount of horizontal pixels that the target takes up in the image
    target_x_image = target_aov * px_per_deg

    # Pixels moved in the image
    motion_blur = target_x_image * percentage_moved

    # print(target_aov)
    print(px_per_deg)

    print("Given that the size of the object is {}m and is moving at the speed of {}ms-1 and that the exposure time is "
          "{}us. ".format(target_x_real, target_speed, exposure_time))
    print("The calculated percentage of the target that has moved in this time is {}.".format(percentage_moved))
    print("The theoretical motion blur is {} pixel".format(motion_blur))

    pass