from Code.process_stream import process_stream
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple program to detect face keypoints')
    parser.add_argument('--source', 
                        type=str, 
                        default='webcam',
                        help='The programs running mode, by default it is webcam. You can pass a path to a videofile to process a video instead',
                        required=False)
    args = parser.parse_args()
    source = args.source
    if source!='webcam' and not os.path.isfile(source):
        print(f'Could not find a file with path: {source}')
    else:
        process_stream(source)