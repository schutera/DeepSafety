# array that lists the names of all classes belonging to the neuronal net
# will be used for printing out the right classname
classnames = ["sign_20","sign_30","sign_50", "sign_60", "sign_70", "sign_80", "sign_80n", "sign_100", "sign_120",
              "overtake_forbidden_pkw", "overtake_forbidden_lkw", "right_of_way", "sign_right_of_way", "sign_give way", "sign_stop",
              "sign_do_not_pass","sign_no_lkw", "sign_wrong_way","sign_danger", "sign_left turn", "sign_right_turn",
              "sharp turn", "sign_uneveness", "sign_slippery", "bottleneck", "sign_construction_work",
              "sign_traffic_light", "sign_pedestrian", "sign_children", "sign_biker", "sign_snow", "sign_animals",
              "sign_no_limit", "sign_drive_right", "sign_drive_left", "sign_drive_straight", "sign_drive_straight_right",
              "sign_drive_straight_left", "sign_obstacle_right", "sign_obstacle_left", "sign_roundabout_traffic",
              "no_overtake_forbidden_pkw", "no_overtake_forbidden_lkw", "human_faces"]

# Definition of the class_detection_function
def class_detection (predictions, ground_truth, classtodetect):
    print('-----------------------------------------------------------') # Layouting
    print('Search for class:', classtodetect)                            # give Information about the regarded class
    print('-----------------------------------------------------------') # Layouting
    print('result:')
    for i in range (0, len(predictions)):                                # loop to check every element of the array
        if predictions[i] == ground_truth[i] == classtodetect:           # first warning
            print(classnames[classtodetect],'found at sample', i, 'WARNING!WARNING!')
        if predictions[i] == classtodetect and ground_truth[i] != classtodetect:    # second warning
            print(classnames[classtodetect],'found at sample', i, 'but isnt. NO WARNING')
        if ground_truth[i]== classtodetect and predictions[i] != classtodetect:         # third warning
            print(classnames[classtodetect],'not detected at sample', i, 'but is WARNING!WARNING!')
    print('-----------------------------------------------------------')    # Layouting