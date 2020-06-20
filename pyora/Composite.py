"""


"""
import numpy as np


"""
Here we implement the Porter-Duff composition operators as required by the ORA spec.
https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators

note that d / b and destination / backdrop are interchangeable

Fa = fraction of the inclusion of the source
Fb = fraction of the inclusion of the destination

co = αs x Fa x Cs + αb x Fb x Cb
"""
#
# def prep_operators(source, destination, opacity, offsets):
#     destination = reshape_dest(destination, source, offsets)
#
#     source_norm = source / 255.0
#     destination_norm = destination / 255.0
#
#     a_s = np.expand_dims(source_norm[:, :, 3], 2) * opacity
#     a_b = np.expand_dims(destination_norm[:, :, 3], 2)
#     c_s = source_norm[:, :, :3]
#     c_b = destination_norm[:, :, :3]
#
#     return a_s, a_b, c_s, c_b
#
# def prep_output(co, ao):
#     c_out = np.dstack((co / ao, ao))
#
#     # co/ao to get color back from calculation
#     # opacity of layer is only applied in the final output alpha
#     np.nan_to_num(c_out, copy=False)
#
#     return c_out * 255.0

def src_over(lower_rgb, upper_rgb):

    return upper_rgb

def plus(lower_rgb, upper_rgb):

    return np.minimum(lower_rgb + upper_rgb, 1.0)


def dst_in(lower_alpha, upper_alpha, lower_rgb, upper_rgb):
    """
    'Clip' composite mode
    All parts of 'layer above' which are alpha in 'layer below' will be made also alpha in 'layer above'
    (to whatever degree of alpha they were)

    Destination which overlaps the source, replaces the source.

    Fa = 0; Fb = αs
    co = αb x Cb x αs
    αo = αb x αs
    """

    out_alpha = lower_alpha * upper_alpha
    out_rgb = np.divide(np.multiply((lower_alpha * upper_alpha)[:, :, None], lower_rgb), out_alpha[:, :, None])

    return out_rgb, out_alpha


def dst_out(lower_alpha, upper_alpha, lower_rgb, upper_rgb):
    """
    reverse 'Clip' composite mode
    All parts of 'layer below' which are alpha in 'layer above' will be made also alpha in 'layer below'
    (to whatever degree of alpha they were)
    """

    out_alpha = lower_alpha * (1 - upper_alpha)
    out_rgb = np.divide(np.multiply((lower_alpha * (1 - upper_alpha))[:, :, None], lower_rgb), out_alpha[:, :, None])

    return out_rgb, out_alpha

def dst_atop(lower_alpha, upper_alpha, lower_rgb, upper_rgb):
    """
    place the layer below above the 'layer above' in places where the 'layer above' exists
    where 'layer below' does not exist, but 'layer above' does, place 'layer-above'
    """

    out_alpha = (upper_alpha * (1 - lower_alpha)) + (lower_alpha * upper_alpha)
    out_rgb = np.divide(
                np.multiply((upper_alpha * (1 - lower_alpha))[:, :, None], upper_rgb) +
                np.multiply((lower_alpha * upper_alpha)[:, :, None], lower_rgb),
                out_alpha[:, :, None]
              )


    return out_rgb, out_alpha



def src_atop(lower_alpha, upper_alpha, lower_rgb, upper_rgb):
    """
    place the layer below above the 'layer above' in places where the 'layer above' exists
    """

    out_alpha = (upper_alpha * lower_alpha) + (lower_alpha * (1 - upper_alpha))
    out_rgb = np.divide(
                np.multiply((upper_alpha * lower_alpha)[:, :, None], upper_rgb) +
                np.multiply((lower_alpha * (1 - upper_alpha))[:, :, None], lower_rgb),
                out_alpha[:, :, None]
              )

    return out_rgb, out_alpha
