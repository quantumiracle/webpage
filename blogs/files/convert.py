from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(mp4_path, gif_path, start_time=None, end_time=None, resize_factor=None):
    """
    Converts an MP4 video file to a GIF.

    Parameters:
    - mp4_path (str): Path to the input MP4 file.
    - gif_path (str): Path to save the output GIF.
    - start_time (float): Start time in seconds for the GIF (optional).
    - end_time (float): End time in seconds for the GIF (optional).
    - resize_factor (float): Factor to resize the video (e.g., 0.5 for 50% smaller, optional).

    Returns:
    - None
    """
    try:
        # Load the video file
        clip = VideoFileClip(mp4_path)

        # Trim the video if start_time and end_time are provided
        if start_time is not None or end_time is not None:
            clip = clip.subclip(start_time, end_time)

        # Resize the video if resize_factor is provided
        if resize_factor is not None:
            clip = clip.resize(resize_factor)

        # Write the video to a GIF
        clip.write_gif(gif_path, fps=15)  # Adjust FPS for smoother GIFs (optional)
        print(f"GIF created successfully: {gif_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
mp4_path = "diffusionforce.mp4"  # Path to your MP4 file
gif_path = "diffusionforcing.gif"       # Path to save the GIF
convert_mp4_to_gif(mp4_path, gif_path, start_time=2, end_time=8, resize_factor=0.5)
