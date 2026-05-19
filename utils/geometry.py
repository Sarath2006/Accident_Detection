def iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes

    :param boxA: (x1, y1, x2, y2)
    :param boxB: (x1, y1, x2, y2)
    :return: IoU value (0 to 1)
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area
