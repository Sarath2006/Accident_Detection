class FrameExtractor:
    """
    Standardizes incoming frames and attaches metadata if needed
    """

    def __init__(self):
        self.frame_id = 0

    def extract(self, frame):
        """
        Returns the frame as-is for now.
        Future:
        - Attach frame_id
        - Attach timestamp
        """
        self.frame_id += 1
        return frame