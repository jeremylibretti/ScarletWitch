import ScarletWitch

def main():
    new_witch = ScarletWitch.ScarletWitch()

    new_witch.run("Game")

def og_main():
    # Argument parsing #################################################################
    #   Capture device, width, and height
    #   Tracking arguments
    #   Use brect

    # Camera preparation ###############################################################
    #   Create video capture device
    #   Set appropriate frame width and height

    # Model load #############################################################
    #   Create mediapipe hands object and point classifiers

    # Read labels ###########################################################
    #   Get csv model labels

    # FPS Measurement ########################################################
    #   Get fps

    # Coordinate history #################################################################
    #   Create queue to track finger point history
    history_length = 16

    # Finger gesture history ################################################
    #   Create queue to track finger gesture history

    #  ########################################################################
    #   Set mode to 0

    # Hand position tracking #####################################################
    #   Create queue to track hand posiiton history

    # Loop until program stops
    #   Get fps
    #
    #   Get key input
    #   If key is ESC, stop program
    #   Use key input to set gesture number and mode
    #
    #   Get camera capture
    #
    #   Get landmarks from image

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # 0: Open
                # 1: Closed
                # 2: Pointer
                # 3: OK
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                
                if not position_tracking and hand_sign_id == 1: # Closed hand
                    position_tracking = True
                elif position_tracking and hand_sign_id != 1:
                    position_tracking = False
                    hand_history.clear()

                if position_tracking:
                    x = (brect[0] + brect[2])/2
                    y = (brect[1] + brect[3])/2
                    hand_history.append([round(x), round(y)])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)

        debug_image = draw_hand_history(debug_image, hand_history)

        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()


# MediaPipe Generating Landmarks
#   -> Landmarks used to determine position and gestures
#       -> Position and gestures used to control game
#       -> Position and gestures displayed on screen