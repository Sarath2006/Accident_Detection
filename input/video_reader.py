import cv2


class VideoReader:
    """
    Handles video input and frame iteration
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"❌ Cannot open video file: {video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        print(f"[INFO] Video loaded: {video_path}")
        print(f"[INFO] FPS: {self.fps}, Total frames: {self.total_frames}")

    def __iter__(self):
        return self

    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration

        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        self.current_frame += 1
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            print("[INFO] Video capture released")

    def get_progress(self):
        """
        Returns progress percentage of video processing
        """
        if self.total_frames == 0:
            return 0
        return round((self.current_frame / self.total_frames) * 100, 2)