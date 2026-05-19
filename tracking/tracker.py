import numpy as np
from utils.geometry import iou


class ObjectTracker:
    """
    Simple IOU-based multi-object tracker
    """

    def __init__(self, max_lost=10):
        self.next_object_id = 0
        self.objects = {}
        self.lost = {}
        self.max_lost = max_lost

    def _register(self, detection):
        self.objects[self.next_object_id] = detection
        self.lost[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.lost[object_id]

    def update(self, detections):
        """
        Updates tracked objects with new detections
        """
        if len(detections) == 0:
            for object_id in list(self.lost.keys()):
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self._deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return self.objects

        object_ids = list(self.objects.keys())
        object_boxes = [self.objects[obj_id]["bbox"] for obj_id in object_ids]

        matched = set()

        for det in detections:
            best_iou = 0
            best_id = None

            for obj_id, box in zip(object_ids, object_boxes):
                overlap = iou(det["bbox"], box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = obj_id

            if best_iou > 0.3:
                self.objects[best_id] = det
                self.lost[best_id] = 0
                matched.add(best_id)
            else:
                self._register(det)

        for obj_id in list(self.objects.keys()):
            if obj_id not in matched:
                self.lost[obj_id] += 1
                if self.lost[obj_id] > self.max_lost:
                    self._deregister(obj_id)

        return self.objects
