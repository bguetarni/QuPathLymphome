import numpy as np

def transform_coordinates(wsi, p, source_level, target_level):
    """
    args:
        wsi (openslide.OpenSlide): slide file
        p (tuple of ints): coordinates in the form (x,y)
        source_level (int): source level of input coordinates
        target_level (int): target level of input coordinates
    """

    # coordinates
    x, y = p

    # source level dimensions
    source_w, source_h = wsi.level_dimensions[source_level]
    
    # target level dimensions
    target_w, target_h = wsi.level_dimensions[target_level]
    
    # scale coordinates
    x = int(x * (target_w/source_w))
    y = int(y * (target_h/source_h))
    
    return x, y

def calculate_region_size(wsi, p0, p1, target_level=None):
    """
    args:
        wsi (openslide.OpenSlide): slide file
        p0 (tuple of ints): upper-left coordinates in the form (x,y)
        p1 (tuple of ints): bottom-right coordinates in the form (x,y)
        target_level (int): target level if area size in another level than 0
    """

    if target_level is not None:
        p0 = transform_coordinates(wsi, p0, 0, target_level)
        p1 = transform_coordinates(wsi, p1, 0, target_level)

    w = p1[0] - p0[0]
    h = p1[1] - p0[1]
    return w, h
