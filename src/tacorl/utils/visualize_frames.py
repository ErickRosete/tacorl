import cv2
import numpy as np

from tacorl.utils.path import get_file_list


def save_video(video_filename: str, images: np.ndarray, fps: int = 45):
    """
    Saves rollout video
    images: np.array, images used to create the video
            shape - seq, height, width, channels
    video_filename: str, path used to saved the video file
    """
    output_video = cv2.VideoWriter(
        video_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        images.shape[1:3][::-1],
    )

    for img in images:
        output_video.write(img)

    output_video.release()


def main():
    dir = "D:/thesis/dataset/rw_play_data_video"
    # dir = "/mnt/SSD/thesis/dataset/real_world_15"
    dir_file_list = get_file_list(dir, sort_list=True)
    rgb_static_images, rgb_gripper_images = [], []
    for i, filepath in enumerate(dir_file_list):
        # print(str(filepaths))
        file = np.load(filepath)
        rgb_static_images.append(file["rgb_static"])
        rgb_gripper_images.append(file["rgb_gripper"])

        if i == 0:
            print("rgb_static shape is", file["rgb_static"].shape)
            print("rgb_gripper shape is", file["rgb_gripper"].shape)

        cv2.imshow("rgb_static", file["rgb_static"][..., ::-1])
        cv2.imshow("rgb_gripper", file["rgb_gripper"][..., ::-1])
        cv2.waitKey(1)

    save_video("rgb_static.mp4", np.stack(rgb_static_images)[..., ::-1])
    save_video("rgb_gripper.mp4", np.stack(rgb_gripper_images)[..., ::-1])


if __name__ == "__main__":
    main()
