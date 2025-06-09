import imageio

def create_video(frame_arrays, filename, fps=5):
    """
    Create a video from a list of frame arrays and save it to the specified filename.

    Parameters:
        frame_arrays (list of np.ndarray): List of frames, where each frame is a numpy array of shape (H, W, C) and dtype uint8.
        filename (str): The output filename for the video.
        fps (int): Frames per second for the video. Default is 5.

    Returns:
        None
    """
    output_path = filename
    writer = imageio.get_writer(
        output_path, fps=fps, codec='libx264', macro_block_size=None
        )
    for frame in frame_arrays:
        assert frame.dtype == np.uint8
        writer.append_data(frame)
    writer.close()


def ordering(filename):
    """
    Given a filename with MetaVQA indexing(for example, 'rgb_5_55.png'), this will
    return the scene-id (first number) and frame-id (second number).
    Parameters:
        filename (str): The filename to parse, expected to be in the format 'rgb_<scene_id>_<frame_id>.png'.
    Returns:
        Tuple: A tuple containing the scene_id and frame_id as integers.
    """
    basename = filename.split(".")[0]
    basename = basename.split("_")
    scene_id, frame_id = int(basename[-2]), int(basename[-1])
    return scene_id, frame_id


if __name__ == "__main__":
    # A sample code to test the create_video function
    # (1) Let's visualize a nuScenes episode.
    import PIL.Image
    import numpy as np
    from glob import glob
    pattern = '*/rgb_*.png'
    scene = "scene-0517_0_39"
    files = glob(f"metavqa_asset/scenarios/nusc_real/{scene}/{pattern}")
    ordered_images = sorted(files, key=ordering)
    frames = [PIL.Image.open(f) for f in ordered_images]
    frame_arrays = [np.array(frame) for frame in frames]
    create_video(frame_arrays, f"metavqa_asset/{scene}_rgb.mp4", fps=2)
    # (2) Let's visualize it from the instance segmentation.
    # Right now, the instanace segmentation massk is not temporally consistent for nuScenes. This will be added in the fuwwture.
    pattern = '*/mask_*.png'
    files = glob(f"metavqa_asset/scenarios/nusc_real/{scene}/{pattern}")
    ordered_images = sorted(files, key=ordering)
    frames = [PIL.Image.open(f) for f in ordered_images]
    frame_arrays = [np.array(frame) for frame in frames]
    create_video(frame_arrays, f"metavqa_asset/{scene}_instance.mp4", fps=2)


