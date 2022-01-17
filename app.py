# Program Wrapper

import scarletwitch
import argparse

import collect_static
import collect_dynamic

def main():
    # Argument parsing
    args = get_args()

    # ScarletWitch creation
    device = scarletwitch.ScarletWitch(args)

    if args.collect_static:
        device = collect_static.StaticDataCollection(args)
    elif args.collect_dynamic:
        device = collect_dynamic.DynamicDataCollection(args)

    # ScarletWitch run
    device.run()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=854)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    parser.add_argument('--collect_static', action='store_true')
    parser.add_argument('--collect_dynamic', action='store_true')
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

if __name__ == '__main__':
    main()