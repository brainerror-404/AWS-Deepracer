import math


def reward_function(params):
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    abs_steering = abs(params['steering_angle'])  # Only need the absolute steering angle
    speed = params['speed']
    progress = params['progress']
    steps = params['steps']
    is_offtrack = params['is_offtrack']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    is_left_of_center = params['is_left_of_center']
    center_variance = params['distance_from_center']/params['track_width']

    # racing line
    left_lane = [38,39,40,41,42,43,44,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,
                 104,105,106,107,108,109,110,111,112,113]
    center_lane = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,27,28,29,30,31,32,33,34,35,
                   36,37,45,46,47,48,49,50,58,59,60,61,62,84,85,
                   86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,114,115,116,117,118]
    right_lane = [22,23,24,25,26,51,52,53,54,55,56,57]

    # speed
    fast = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,36,37,38,39,40,41,42,43,
            44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,87,88,
            89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118]
    slow = [31,32,33,34,35,79,80,81,82,83,84,85,86]

    # Set benchmark
    benchmark_time = 10.0
    benchmark_steps = 140

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # initialize reward
    reward = 1.0

    # Give higher reward if the car is closer to center line and on track
    if distance_from_center <= marker_1 and all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.05:
        reward += 1.0
    elif distance_from_center <= marker_2 and all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.1:
        reward += 0.5
    elif distance_from_center <= marker_3 and (0.5 * track_width - distance_from_center) >= 0.5:
        reward += 0.1
    else:
        reward *= 0.5 # likely crashed/ close to off track

    # Give higher reward if car finish track
    if progress == 100:
        if round(steps/15,1) < benchmark_time:
            reward *= 1.5
        else:
            reward += 1.0
    elif is_offtrack:
        reward *= 0.5

    # Give reward if car passes 50 steps faster then expected
    if (steps % 50) == 0 and progress >= (steps/benchmark_steps) * 100:
        reward *= 2.0
    else:
        reward *= 0.5

    # Calculate the direction of center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan(dy,dx), the result is (-pi,pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degrees
    track_direction = math.degrees(track_direction)

    # Steering penalty threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 10
    SPEED_THRESHOLD_FAST = 2.0  # 4.0
    SPEED_THRESHOLD_SLOW = 1.5  # 3.5
    DIRECTION_THRESHOLD = 3.0

    # Calculate the difference between track direction and heading direction
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    if direction_diff > DIRECTION_THRESHOLD and is_offtrack:
        reward *= 0.5
    else:
        reward += 1.0

    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.5
    else:
        reward += 1.0

    # Penalise reward if car not in correct lane
    if closest_waypoints[1] in left_lane and is_left_of_center:
        reward += 10.0
    elif closest_waypoints[1] in right_lane and not is_left_of_center:
        reward += 10.0
    elif closest_waypoints[1] in center_lane and center_variance < 0.4:
        reward += 10.0
    else:
        reward -= 10.0

    # Penalize reward if the car is too slow
    if closest_waypoints[1] in fast:
        if speed >= SPEED_THRESHOLD_FAST:
            reward *= 2.0
        else:
            reward *= 0.5
    elif closest_waypoints[1] in slow:
        if speed >= SPEED_THRESHOLD_SLOW:
            reward *= 2.0
        else:
            reward *= 0.5

    if speed < 0.8:
        reward *= 0.5

    # if car is straight, go faster
    if abs(abs_steering) < 0.1 and speed >= SPEED_THRESHOLD_FAST:
        reward *= 1.5
    elif abs(abs_steering) < 0.25 and speed >= SPEED_THRESHOLD_SLOW:
        reward *= 1.2

    # Self Motivator
    if all_wheels_on_track and steps > 0:
        reward = ((progress/steps)*100) + (speed**2)
    else:
        reward += 1e-3

    return float(reward)
