import utils
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import preprocess
from multiprocessing import Process, cpu_count

def get_words(video_name):
    word_map = {
        0: {
            'b': 'bin',
            'l': 'lay',
            'p': 'place',
            's': 'set',
        },
        1: {
            'b': 'blue',
            'g': 'green',
            'r': 'red',
            'w': 'white',
        },
        2: {
            'a': 'at',
            'b': 'by',
            'i': 'in',
            'w': 'with',
        },
        5: {
            'a': 'again',
            'n': 'now',
            'p': 'please',
            's': 'soon',
        }
    }

    words = []
    for i in range(len(video_name)):
        char = video_name[i]
        if i in [0,1,2,5]:
            words.append(word_map[i][char])
        if i == 3:
            words.append(char)
        if i == 4:
            if char.isdigit():
                words.append(char)
            else:
                words.append('0')

    return ' '.join(words)



def extract_frames_from_video(video_path, output_directory):
    # Create the output directory if it doesn't exist

    frames = []
    frame_filenames = []

    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    speaker_name = os.path.basename(os.path.dirname(video_path))

    output_directory = os.path.join(output_directory,speaker_name, video_name)

    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    words = get_words(video_name)
    # output words to file
    with open(os.path.join(output_directory, 'words.txt'), 'w') as f:
        f.write(words)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_count += 1
        #frame = preprocess.process_image(frame, output_size=(45,25))

        if frame is None:
            continue

        # Save the frame as an image
        frame_filename = os.path.join(output_directory, f"frame_{frame_count:04d}.png")

        frames.append(frame)
        frame_filenames.append(frame_filename)
        
    frames = preprocess_frames(frames)

    for (frame, frame_filename) in zip(frames, frame_filenames):
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")


def preprocess_frames(frames):

    # Convert the frames to grayscale
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    # Mask the lips in the frames
    lip_imgs = preprocess.mask_lips_batch(grays, padding=3, crop=True)

    # Scale the images to 45x25
    lip_imgs = [preprocess.scale(lip_img, output_size=(45,25)) for lip_img in lip_imgs]

    # Apply edge detection to the images
    #lip_imgs = [preprocess.edge_detection(lip_img) for lip_img in lip_imgs]

    return lip_imgs

data_path = ".\\data"
output_directory = ".\\frames"

if __name__ == "__main__":
        
    # Get the paths to all the videos
    video_paths = utils.get_paths(Path(data_path), file_type='.mpg')
    video_paths = utils.filter_paths(video_paths, output_directory)

    # Calculate the number of available CPU cores and use 75% of them
    num_cores = cpu_count()
    num_processes = max(1, int(0.75 * num_cores))

    processes = []

    # Extract the frames from each video
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]

        # Create a process to extract the frames
        process = Process(target=extract_frames_from_video, args=(video_path, output_directory))

        # Start the process
        process.start()

        # Add the process to the list
        processes.append(process)

        # If the number of active processes reaches the desired amount, wait for them to finish
        if len(processes) >= num_processes:
            for p in processes:
                p.join()
            processes = []  # Reset the list

    # Wait for any remaining processes to finish
    for process in processes:
        process.join()