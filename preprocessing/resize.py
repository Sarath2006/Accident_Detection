import cv2

class FrameResizer:
    """
    Resizes frames to a fixed size for model compatibility
    """

    def __init__(self, target_size):
        """
        :param target_size: (width, height)
        """
        self.target_width = target_size[0]
        self.target_height = target_size[1]

    def apply(self, frame):
        """
        Resize the frame to the target size
        """
        if frame is None:
            return None

        resized_frame = cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )
        return resized_frame