import numpy as np
from pyora.Composite import dst_in, dst_out, src_atop, dst_atop

no_alpha_blend_funcs = [dst_in, dst_out, src_atop, dst_atop]

def outer_shell(lower, upper, opacity, offsets, blend_func):
    # do any offset shifting first
    if offsets[0] > 0:
        upper = np.hstack((np.zeros((upper.shape[0], offsets[0], 4), dtype=np.float64), upper))
    elif offsets[0] < 0:
        if offsets[0] > -1 * upper.shape[1]:
            upper = upper[:, -1 * offsets[0]:, :]
        else:
            # offset offscreen completely, there is nothing left
            return np.zeros(lower.shape, dtype=np.float64)
    if offsets[1] > 0:
        upper = np.vstack((np.zeros((offsets[1], upper.shape[1], 4), dtype=np.float64), upper))
    elif offsets[1] < 0:
        if offsets[1] > -1 * upper.shape[0]:
            upper = upper[-1 * offsets[1]:, :, :]
        else:
            # offset offscreen completely, there is nothing left
            return np.zeros(lower.shape, dtype=np.float64)

    # resize array to fill small images with zeros
    if upper.shape[0] < lower.shape[0]:
        upper = np.vstack(
            (upper, np.zeros((lower.shape[0] - upper.shape[0], upper.shape[1], 4), dtype=np.float64)))
    if upper.shape[1] < lower.shape[1]:
        upper = np.hstack(
            (upper, np.zeros((upper.shape[0], lower.shape[1] - upper.shape[1], 4), dtype=np.float64)))

    # crop the source if the backdrop is smaller
    upper = upper[:lower.shape[0], :lower.shape[1], :]

    lower_norm = lower / 255.0
    upper_norm = upper / 255.0

    upper_alpha = upper_norm[:, :, 3] * opacity
    lower_alpha = lower_norm[:, :, 3]

    upper_rgb = upper_norm[:, :, :3]
    lower_rgb = lower_norm[:, :, :3]

    if blend_func in no_alpha_blend_funcs:
        out_rgb, out_alpha = blend_func(lower_alpha, upper_alpha, lower_rgb, upper_rgb)
    else:
        out_rgb, out_alpha = alpha_comp_shell(lower_alpha, upper_alpha, lower_rgb, upper_rgb, blend_func)

    out = np.nan_to_num(np.dstack((out_rgb, out_alpha,)), copy=False) * 255.0


    return out

def alpha_comp_shell(lower_alpha, upper_alpha, lower_rgb, upper_rgb, blend_func):
    """
    Common transformations that will occur with any blend or composite mode 
    """


    out_alpha = upper_alpha + lower_alpha - (upper_alpha * lower_alpha)

    blend_rgb = blend_func(lower_rgb, upper_rgb)

    lower_rgb_part = np.multiply(((1.0 - upper_alpha) * lower_alpha)[:, :, None], lower_rgb)
    upper_rgb_part = np.multiply(((1.0 - lower_alpha) * upper_alpha)[:, :, None], upper_rgb)
    blended_rgb_part = np.multiply((lower_alpha * upper_alpha)[:, :, None], blend_rgb)

    out_rgb = np.divide((lower_rgb_part + upper_rgb_part + blended_rgb_part), out_alpha[:, :, None])

    return out_rgb, out_alpha

#
# def _compose_alpha(source, destination, opacity, window=None):
#     """Calculate alpha composition ratio between two images.
#     window: tuple of min_x, max_x, min_y, max_y to actually blend
#     """
#
#     comp_alpha = np.minimum(destination[:, :, 3], source[:, :, 3]) * opacity
#     new_alpha = destination[:, :, 3] + (1.0 - destination[:, :, 3]) * comp_alpha
#     np.seterr(divide='ignore', invalid='ignore')
#     ratio = comp_alpha / new_alpha
#     ratio[ratio == np.NAN] = 0.0
#
#
#     # make sure to get a full mask on parts of the image which are not part of both source and backdrop
#     if window:
#         mask = np.ones_like(ratio, dtype=bool)
#         mask[window[0]:window[1], window[2]:window[3]] = False
#         ratio[mask] = 1.0
#     return ratio

# def reshape_dest(source, destination, offsets):
#     # shift destination by offset if needed by adding rows and cols of zeros before
# 
#     if offsets[0] > 0:
#         source = np.hstack((np.zeros((source.shape[0], offsets[0], 4), dtype=np.float64), source))
#     elif offsets[0] < 0:
#         if offsets[0] > -1*source.shape[1]:
#             source = source[:, -1 * offsets[0]:, :]
#         else:
#             # offset offscreen completely, there is nothing left
#             return np.zeros(destination.shape, dtype=np.float64)
#     if offsets[1] > 0:
#         source = np.vstack((np.zeros((offsets[1], source.shape[1], 4), dtype=np.float64), source))
#     elif offsets[1] < 0:
#         if offsets[1] > -1 * source.shape[0]:
#             source = source[-1 * offsets[1]:, :, :]
#         else:
#             # offset offscreen completely, there is nothing left
#             return np.zeros(destination.shape, dtype=np.float64)
# 
# 
#     # resize array to fill small images with zeros
#     if source.shape[0] < destination.shape[0]:
#         source = np.vstack(
#             (source, np.zeros((destination.shape[0] - source.shape[0], source.shape[1], 4), dtype=np.float64)))
#     if source.shape[1] < destination.shape[1]:
#         source = np.hstack(
#             (source, np.zeros((source.shape[0], destination.shape[1] - source.shape[1], 4), dtype=np.float64)))
# 
#     # crop the source if the backdrop is smaller
#     source = source[:destination.shape[0], :destination.shape[1], :]
# 
#     return source
# 
# 
# def normal(source, destination, opacity, offsets=(0, 0)):
#     """Apply "normal" blending mode of a layer on an image.
#     """
# 
#     source = reshape_dest(source, destination, offsets)
# 
#     destination_norm = destination / 255.0
#     source_norm = source / 255.0
# 
#     # Extract alpha-channels and apply opacity
#     destination_alp = np.expand_dims(destination_norm[:, :, 3], 2)  # alpha of b, prepared for broadcasting
#     source_alp = np.expand_dims(source_norm[:, :, 3], 2) * opacity  # alpha of a, prepared for broadcasting
# 
#     # Blend images
# 
#     with np.errstate(invalid='ignore'):
#         c_out = (source_norm[:, :, :3] * source_alp + destination_norm[:, :, :3] * destination_alp * (1 - source_alp)) \
#             / (source_alp + destination_alp * (1 - source_alp))
# 
#     # Blend alpha
#     cout_alp = source_alp + destination_alp * (1 - source_alp)
# 
#     # Combine image and alpha
#     c_out = np.dstack((c_out, cout_alp))
# 
#     np.nan_to_num(c_out, copy=False)
# 
#     return c_out * 255.0
# 
# 
# def _old_blend_mode(source, destination, opacity, offsets=(0, 0)):
#     """
#     Preserved implementation of the blend mode, there opacity of the blend mode actually also mixed
#     the colors of the blended mode and the backdrop.
# 
#     In current ORA standard, porter-duff blend modes are used, which do not mix the post-blended
#     layer with the backdrop. However, this might be an option at some point in the future.
#     """
#     source = reshape_dest(source, destination, offsets)
# 
#     destination_norm = destination / 255.0
#     source_norm = source / 255.0
# 
#     ratio = _compose_alpha(destination_norm, source_norm, opacity)
# 
#     # 'comp' is the actual blend mode
#     comp = np.maximum(destination_norm[:, :, :3], source_norm[:, :, :3])
# 
#     ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
#     img_out = comp * ratio_rs + source_norm[:, :, :3] * (1.0 - ratio_rs)
#     img_out = np.nan_to_num(np.dstack((img_out, source_norm[:, :, 3])))  # add alpha channel and replace nans
#     return img_out * 255.0
# 
# def _general_blend(source, destination, offsets, comp_func):
#     source = reshape_dest(source, destination, offsets)
# 
#     destination_norm = destination / 255.0
#     source_norm = source / 255.0
# 
#     comp = comp_func(destination_norm, source_norm)
# 
#     idxs = destination_norm[:, :, 3] == 0
# 
#     ratio_rs = np.reshape(np.repeat(idxs, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
#     img_out = comp + (source_norm[:, :, :3] * ratio_rs)
#     img_out = np.nan_to_num(np.dstack((img_out, source_norm[:, :, 3])))  # add alpha channel and replace nans
# 
#     return img_out * 255.0
# 


def soft_light(lower_rgb, upper_rgb):
    """Apply soft light blending mode of a layer on an image.
    """
    return (1.0 - lower_rgb) * lower_rgb * upper_rgb \
           + lower_rgb * (1.0 - (1.0 - lower_rgb) * (1.0 - upper_rgb))

def lighten_only(lower_rgb, upper_rgb):
    """Apply lighten only blending mode of a layer on an image.
    """
    return np.maximum(lower_rgb, upper_rgb)

def screen(lower_rgb, upper_rgb):
    """Apply screen blending mode of a layer on an image.

    """
    return 1.0 - (1.0 - lower_rgb) * (1.0 - upper_rgb)

def dodge(lower_rgb, upper_rgb):
    """Apply dodge blending mode of a layer on an image.
    """
    return np.minimum(lower_rgb / ((1.0 + np.finfo(np.float64).eps) - upper_rgb), 1.0)

def burn(lower_rgb, upper_rgb):
    """Apply burn blending mode of a layer on an image.
    """
    return np.maximum(1.0 - (((1.0 + np.finfo(np.float64).eps) - lower_rgb) / upper_rgb), 0.0)

def addition(lower_rgb, upper_rgb):
    """Apply addition blending mode of a layer on an image.
    """
    return lower_rgb + upper_rgb

def darken_only(lower_rgb, upper_rgb):
    """Apply darken only blending mode of a layer on an image.
    """
    return np.minimum(lower_rgb, upper_rgb)

def multiply(lower_rgb, upper_rgb):
    """Apply multiply blending mode of a layer on an image.
    """
    return np.clip(upper_rgb * lower_rgb, 0.0, 1.0)

def hard_light(lower_rgb, upper_rgb):
    """Apply hard light blending mode of a layer on an image.
    """
    return np.greater(upper_rgb, 0.5) \
           * np.minimum(1.0 - ((1.0 - lower_rgb)
                               * (1.0 - (upper_rgb - 0.5) * 2.0)), 1.0) \
           + np.logical_not(np.greater(upper_rgb, 0.5)) \
           * np.minimum(lower_rgb * (upper_rgb * 2.0), 1.0)

def difference(lower_rgb, upper_rgb):
    """Apply difference blending mode of a layer on an image.
    """
    comp = lower_rgb - upper_rgb
    comp[comp < 0.0] *= -1.0
    return comp

def subtract(lower_rgb, upper_rgb):
    """Apply subtract blending mode of a layer on an image.
    """
    return upper_rgb - upper_rgb

def grain_extract(lower_rgb, upper_rgb):
    """Apply grain extract blending mode of a layer on an image.
    """
    return np.clip(lower_rgb - upper_rgb + 0.5, 0.0, 1.0)

def grain_merge(lower_rgb, upper_rgb):
    """Apply grain merge blending mode of a layer on an image.
    """
    return np.clip(lower_rgb + upper_rgb - 0.5, 0.0, 1.0)

def divide(lower_rgb, upper_rgb):
    """Apply divide blending mode of a layer on an image.
    """
    return np.minimum((256.0 / 255.0 * lower_rgb) / (1.0 / 255.0 + upper_rgb), 1.0)


def overlay(lower_rgb, upper_rgb):
    """Apply overlay blending mode of a layer on an image.
    """
    return np.less(lower_rgb, 0.5) * (2 * lower_rgb * upper_rgb) \
           + np.greater_equal(lower_rgb, 0.5) \
           * (1 - (2 * (1 - lower_rgb) * (1 - upper_rgb)))

    
