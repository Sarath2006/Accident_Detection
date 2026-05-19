import math
from collections import defaultdict, deque

class SpeedEstimator:
    """
    Estimates relative speed of tracked objects using bounding box movement
    """

    def __init__(self, history_size=5):
        self.history_size = history_size
        self.centroid_history = defaultdict(lambda: deque(maxlen=self.history_size))

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def compute(self, tracked_objects):
        """
        Computes relative speed for each tracked object

        :param tracked_objects: dict of tracked objects
        :return: dict of object_id -> speed
        """
        speeds = {}

        for object_id, obj in tracked_objects.items():
            bbox = obj["bbox"]
            centroid = self._centroid(bbox)
            self.centroid_history[object_id].append(centroid)

            if len(self.centroid_history[object_id]) >= 2:
                (x1, y1), (x2, y2) = \
                    self.centroid_history[object_id][-2], \
                    self.centroid_history[object_id][-1]

                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                speeds[object_id] = distance
            else:
                speeds[object_id] = 0.0

        return speeds