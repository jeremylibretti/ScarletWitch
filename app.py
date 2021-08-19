import ScarletWitch
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--use_training_mode', action='store_true')
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def main():
    # Argument parsing #################################################################
    args = get_args()

    training = args.use_training_mode

    # if training:
    #     device = ScarletWitch.ScarletWitch()
    # else:
    device = ScarletWitch.ScarletWitch(args)

    device.run()

if __name__ == '__main__':
    main()

# MediaPipe Generating Landmarks
#   -> Landmarks used to determine position and gestures
#       -> Position and gestures used to control game
#       -> Position and gestures displayed on screen