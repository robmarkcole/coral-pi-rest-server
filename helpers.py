"""
Helper functions.
"""


def get_object_classes(predictions, confidence):
    """
    Get a list of the unique object classes predicted.
    """
    classes = [
        pred["label"] for pred in predictions if float(pred["confidence"]) >= confidence
    ]
    return set(classes)


def get_object_instances(predictions, target, confidence):
    """
    Return the number of instances of a target class.
    """
    targets_identified = [
        pred
        for pred in predictions
        if pred["label"] == target and float(pred["confidence"]) >= confidence
    ]
    return len(targets_identified)


def get_objects_summary(predictions, confidence):
    """
    Get a summary of the objects detected.
    """
    classes = get_object_classes(predictions, confidence)
    return {
        label: get_object_instances(predictions, label, confidence) for label in classes
    }
