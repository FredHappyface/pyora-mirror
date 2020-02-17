import numpy as np



def _compose_alpha(source, destination, opacity, window=None):
    """Calculate alpha composition ratio between two images.
    window: tuple of min_x, max_x, min_y, max_y to actually blend
    """

    comp_alpha = np.minimum(destination[:, :, 3], source[:, :, 3]) * opacity
    new_alpha = destination[:, :, 3] + (1.0 - destination[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.NAN] = 0.0


    # make sure to get a full mask on parts of the image which are not part of both source and backdrop
    if window:
        mask = np.ones_like(ratio, dtype=bool)
        mask[window[0]:window[1], window[2]:window[3]] = False
        ratio[mask] = 1.0
    return ratio

def reshape_dest(source, destination, offsets):
    # shift destination by offset if needed by adding rows and cols of zeros before

    if offsets[0] > 0:
        source = np.hstack((np.zeros((source.shape[0], offsets[0], 4), dtype=np.float64), source))
    elif offsets[0] < 0:
        if offsets[0] > -1*source.shape[1]:
            source = source[:, -1 * offsets[0]:, :]
        else:
            # offset offscreen completely, there is nothing left
            return np.zeros(destination.shape, dtype=np.float64)
    if offsets[1] > 0:
        source = np.vstack((np.zeros((offsets[1], source.shape[1], 4), dtype=np.float64), source))
    elif offsets[1] < 0:
        if offsets[1] > -1 * source.shape[0]:
            source = source[-1 * offsets[1]:, :, :]
        else:
            # offset offscreen completely, there is nothing left
            return np.zeros(destination.shape, dtype=np.float64)


    # resize array to fill small images with zeros
    if source.shape[0] < destination.shape[0]:
        source = np.vstack(
            (source, np.zeros((destination.shape[0] - source.shape[0], source.shape[1], 4), dtype=np.float64)))
    if source.shape[1] < destination.shape[1]:
        source = np.hstack(
            (source, np.zeros((source.shape[0], destination.shape[1] - source.shape[1], 4), dtype=np.float64)))

    # crop the source if the backdrop is smaller
    source = source[:destination.shape[0], :destination.shape[1], :]

    return source


def normal(source, destination, opacity, offsets=(0, 0)):
    """Apply "normal" blending mode of a layer on an image.
    """

    source = reshape_dest(source, destination, offsets)

    destination_norm = destination / 255.0
    source_norm = source / 255.0

    # Extract alpha-channels and apply opacity
    destination_alp = np.expand_dims(destination_norm[:, :, 3], 2)  # alpha of b, prepared for broadcasting
    source_alp = np.expand_dims(source_norm[:, :, 3], 2) * opacity  # alpha of a, prepared for broadcasting

    # Blend images

    with np.errstate(invalid='ignore'):
        c_out = (source_norm[:, :, :3] * source_alp + destination_norm[:, :, :3] * destination_alp * (1 - source_alp)) \
            / (source_alp + destination_alp * (1 - source_alp))

    # Blend alpha
    cout_alp = source_alp + destination_alp * (1 - source_alp)

    # Combine image and alpha
    c_out = np.dstack((c_out, cout_alp))

    np.nan_to_num(c_out, copy=False)

    return c_out * 255.0


def _old_blend_mode(source, destination, opacity, offsets=(0, 0)):
    """
    Preserved implementation of the blend mode, there opacity of the blend mode actually also mixed
    the colors of the blended mode and the backdrop.

    In current ORA standard, porter-duff blend modes are used, which do not mix the post-blended
    layer with the backdrop. However, this might be an option at some point in the future.
    """
    source = reshape_dest(source, destination, offsets)

    destination_norm = destination / 255.0
    source_norm = source / 255.0

    ratio = _compose_alpha(destination_norm, source_norm, opacity)

    # 'comp' is the actual blend mode
    comp = np.maximum(destination_norm[:, :, :3], source_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + source_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, source_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0

def _general_blend(source, destination, offsets, comp_func):
    source = reshape_dest(source, destination, offsets)

    destination_norm = destination / 255.0
    source_norm = source / 255.0

    comp = comp_func(destination_norm, source_norm)

    idxs = destination_norm[:, :, 3] == 0

    ratio_rs = np.reshape(np.repeat(idxs, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp + (source_norm[:, :, :3] * ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, source_norm[:, :, 3])))  # add alpha channel and replace nans

    return img_out * 255.0



def soft_light(source, destination, offsets=(0, 0)):
    """Apply soft light blending mode of a layer on an image.
    """
    def comp_func(destination_norm, source_norm):
        return (1.0 - destination_norm[:, :, :3]) * destination_norm[:, :, :3] * source_norm[:, :, :3] \
           + destination_norm[:, :, :3] * (1.0 - (1.0 - destination_norm[:, :, :3]) * (1.0 - source_norm[:, :, :3]))

    return _general_blend(source, destination, offsets, comp_func)


def lighten_only(source, destination, offsets=(0, 0)):
    """Apply lighten only blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.maximum(destination_norm[:, :, :3], source_norm[:, :, :3])

    return _general_blend(source, destination, offsets, comp_func)

def screen(source, destination, offsets=(0, 0)):
    """Apply screen blending mode of a layer on an image.

    """

    def comp_func(destination_norm, source_norm):
        return 1.0 - (1.0 - destination_norm[:, :, :3]) * (1.0 - source_norm[:, :, :3])

    return _general_blend(source, destination, offsets, comp_func)

def dodge(source, destination, offsets=(0, 0)):
    """Apply dodge blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.minimum(destination_norm[:, :, :3] / ((1.0 + np.finfo(np.float64).eps) - source_norm[:, :, :3]), 1.0)

    return _general_blend(source, destination, offsets, comp_func)

def burn(source, destination, offsets=(0, 0)):
    """Apply burn blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.maximum(1.0 - (((1.0 + np.finfo(np.float64).eps) - destination_norm[:, :, :3]) / source_norm[:, :, :3]), 0.0)

    return _general_blend(source, destination, offsets, comp_func)

def addition(source, destination, offsets=(0, 0)):
    """Apply addition blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return destination_norm[:, :, :3] + source_norm[:, :, :3]

    return _general_blend(source, destination, offsets, comp_func)

def darken_only(source, destination, offsets=(0, 0)):
    """Apply darken only blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.minimum(destination_norm[:, :, :3], source_norm[:, :, :3])

    return _general_blend(source, destination, offsets, comp_func)

def multiply(source, destination, offsets=(0, 0)):
    """Apply multiply blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.clip(source_norm[:, :, :3] * destination_norm[:, :, :3], 0.0, 1.0)

    return _general_blend(source, destination, offsets, comp_func)

def hard_light(source, destination, offsets=(0, 0)):
    """Apply hard light blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.greater(source_norm[:, :, :3], 0.5) \
           * np.minimum(1.0 - ((1.0 - destination_norm[:, :, :3])
                               * (1.0 - (source_norm[:, :, :3] - 0.5) * 2.0)), 1.0) \
           + np.logical_not(np.greater(source_norm[:, :, :3], 0.5)) \
           * np.minimum(destination_norm[:, :, :3] * (source_norm[:, :, :3] * 2.0), 1.0)

    return _general_blend(source, destination, offsets, comp_func)

def difference(source, destination, offsets=(0, 0)):
    """Apply difference blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        comp = destination_norm[:, :, :3] - source_norm[:, :, :3]
        comp[comp < 0.0] *= -1.0
        return comp

    return _general_blend(source, destination, offsets, comp_func)

def subtract(source, destination, offsets=(0, 0)):
    """Apply subtract blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return destination[:, :, :3] - source_norm[:, :, :3]

    return _general_blend(source, destination, offsets, comp_func)

def grain_extract(source, destination, offsets=(0, 0)):
    """Apply grain extract blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.clip(destination_norm[:, :, :3] - source_norm[:, :, :3] + 0.5, 0.0, 1.0)

    return _general_blend(source, destination, offsets, comp_func)

def grain_merge(source, destination, offsets=(0, 0)):
    """Apply grain merge blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.clip(destination_norm[:, :, :3] + source_norm[:, :, :3] - 0.5, 0.0, 1.0)

    return _general_blend(source, destination, offsets, comp_func)

def divide(source, destination, offsets=(0, 0)):
    """Apply divide blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.minimum((256.0 / 255.0 * destination_norm[:, :, :3]) / (1.0 / 255.0 + source_norm[:, :, :3]), 1.0)

    return _general_blend(source, destination, offsets, comp_func)

def overlay(source, destination,  offsets=(0, 0)):
    """Apply overlay blending mode of a layer on an image.
    """

    def comp_func(destination_norm, source_norm):
        return np.less(destination_norm[:, :, :3], 0.5) * (2 * destination_norm[:, :, :3] * source_norm[:, :, :3]) \
           + np.greater_equal(destination_norm[:, :, :3], 0.5) \
           * (1 - (2 * (1 - destination_norm[:, :, :3]) * (1 - source_norm[:, :, :3])))

    return _general_blend(source, destination, offsets, comp_func)