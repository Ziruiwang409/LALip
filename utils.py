import os
import cv2

def get_paths(data_path, file_type='.mpg'):
    video_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(file_type):
                video_paths.append(os.path.join(root, file))
    return video_paths



def filter_paths(video_paths, output_directory):
    filtered_paths = []

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        speaker_name = os.path.basename(os.path.dirname(video_path))
        output_dir = os.path.join(output_directory, speaker_name, video_name)

        # Check if the output directory exists and contains at least 75 items
        if not os.path.exists(output_dir) or len(os.listdir(output_dir)) < 75:
            filtered_paths.append(video_path)

    return filtered_paths

    

   
            
        
