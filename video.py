import os
import cv2

def create_video_from_images(folder_path, output_path, fps=25):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])

    if not image_files:
        print("No image files found in the folder.")
        return

    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    frame = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, channels = frame.shape

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(os.path.join(folder_path, image_file))
        video_writer.write(frame)

    video_writer.release()
    print("Video created successfully.")

# Example usage
folder_path = 'heat/'  # Replace with the path to your image folder
output_path = 'heat.mp4'  # Replace with the desired output video path
fps = 25  # Frames per second (optional, default is 25)

create_video_from_images(folder_path, output_path, fps)

