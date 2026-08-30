"""Apply Qwen pixel bounds to both fast and legacy image processors."""
import copy
from contextlib import contextmanager


def configure_visual_pixel_limit(processor, limit):
    if limit is None:
        return
    image = getattr(processor, "image_processor", None)
    if image is None or type(limit) is not int or limit <= 0:
        raise ValueError("visual pixel limit requires an image processor and positive integer")
    patch = getattr(image, "patch_size", 1)
    merge = getattr(image, "merge_size", 1)
    if type(patch) is int and type(merge) is int and limit < (patch * merge) ** 2:
        raise ValueError("visual pixel limit is smaller than one merged visual patch")
    size = getattr(image, "size", None)
    if isinstance(size, dict) and {"shortest_edge", "longest_edge"} <= size.keys():
        minimum = min(size["shortest_edge"], limit)
        # FastImageProcessor 从 size 或成对 min/max kwargs 派生尺寸；单独设置 max 会失效。
        image.size = {**size, "shortest_edge": minimum, "longest_edge": limit}
        image.min_pixels = minimum
    elif hasattr(image, "max_pixels"):
        minimum = getattr(image, "min_pixels", None)
        if minimum is not None:
            image.min_pixels = min(minimum, limit)
    else:
        raise ValueError("image processor does not expose supported visual pixel bounds")
    image.max_pixels = limit


@contextmanager
def temporary_visual_pixel_limit(processor, limit):
    """实验组只临时改处理器；异常也恢复，且调用者须持有模型执行锁。"""
    if limit is None:
        yield
        return
    image = getattr(processor, "image_processor", None)
    original = {key: copy.deepcopy(getattr(image, key)) for key in ("size", "min_pixels", "max_pixels")
                if hasattr(image, key)}
    try:
        configure_visual_pixel_limit(processor, limit)
        yield
    finally:
        for key in ("size", "min_pixels", "max_pixels"):
            if key in original:
                setattr(image, key, original[key])
            elif hasattr(image, key):
                delattr(image, key)
