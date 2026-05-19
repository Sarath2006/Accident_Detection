import cv2


class FrameDenoiser:
    """
    Applies noise reduction to frames
    """

    def __init__(self, method="gaussian"):
        """
        :param method: 'gaussian' or 'median'
        """
        self.method = method

    def apply(self, frame):
        if frame is None:
            return None

        if self.method == "gaussian":
            # Gaussian blur (good balance of smoothing + edge preservation)
            return cv2.GaussianBlur(frame, (5, 5), 0)

        elif self.method == "median":
            # Median blur (good for salt-and-pepper noise)
            return cv2.medianBlur(frame, 5)

        # Fallback (no denoising)
        return frame
