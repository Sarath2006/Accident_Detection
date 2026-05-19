import cv2
import numpy as np


class VideoStabilizer:
    """
    Applies basic video stabilization using optical flow
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_frame = None

    def apply(self, frame):
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame, nothing to stabilize
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame
            return frame

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        # Estimate global motion
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])

        # Build transformation matrix
        transform = np.float32([[1, 0, -dx], [0, 1, -dy]])

        stabilized = cv2.warpAffine(
            frame,
            transform,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        self.prev_gray = gray
        self.prev_frame = frame

        return stabilized
