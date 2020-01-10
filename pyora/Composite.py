"""


"""
import numpy as np
from pyora.Blend import reshape_dest


"""
Here we implement the Porter-Duff composition operators as required by the ORA spec.
https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators

note that d / b and destination / backdrop are interchangeable

Fa = fraction of the inclusion of the source
Fb = fraction of the inclusion of the destination

co = αs x Fa x Cs + αb x Fb x Cb
"""

def prep_operators(source, destination, opacity, offsets):
    destination = reshape_dest(destination, source, offsets)

    source_norm = source / 255.0
    destination_norm = destination / 255.0

    a_s = np.expand_dims(source_norm[:, :, 3], 2) * opacity
    a_b = np.expand_dims(destination_norm[:, :, 3], 2)
    c_s = source_norm[:, :, :3]
    c_b = destination_norm[:, :, :3]

    return a_s, a_b, c_s, c_b

def prep_output(co, ao):
    c_out = np.dstack((co / ao, ao))

    # co/ao to get color back from calculation
    # opacity of layer is only applied in the final output alpha
    np.nan_to_num(c_out, copy=False)

    return c_out * 255.0

def dst_in(source, destination, opacity=1.0, offsets=(0, 0)):
    """
    'Clip' composite mode
    All parts of 'layer above' which are alpha in 'layer below' will be made also alpha in 'layer above'
    (to whatever degree of alpha they were)

    Destination which overlaps the source, replaces the source.

    Fa = 0; Fb = αs
    co = αb x Cb x αs
    αo = αb x αs

    :param source:
    :param destination:
    :return:
    """

    a_s, a_b, c_s, c_b = prep_operators(source, destination, opacity, offsets)

    co = a_b * c_b * a_s
    ao = a_b * a_s

    return prep_output(co, ao)


def dst_out(source, destination, opacity=1.0, offsets=(0, 0)):
    """
    reverse 'Clip' composite mode
    All parts of 'layer below' which are alpha in 'layer above' will be made also alpha in 'layer below'
    (to whatever degree of alpha they were)
    :param img_in:
    :param img_layer:
    :return:
    """
    a_s, a_b, c_s, c_b = prep_operators(source, destination, opacity, offsets)

    co = a_b * c_b * (1 - a_s)
    ao = a_b * (1 - a_s)

    return prep_output(co, ao)

def dst_atop(source, destination, opacity=1.0, offsets=(0, 0)):
    """
    place the layer below above the 'layer above' in places where the 'layer above' exists
    where 'layer below' does not exist, but 'layer above' does, place 'layer-above'

    :param img_in:
    :param img_layer:
    :return:
    """

    a_s, a_b, c_s, c_b = prep_operators(source, destination, opacity, offsets)

    co = (a_s * c_s * (1 - a_b)) + (a_b * c_b * a_s)
    ao = (a_s * (1 - a_b)) + (a_b * a_s)

    return prep_output(co, ao)



def src_atop(source, destination, opacity=1.0, offsets=(0, 0)):
    """
    place the layer below above the 'layer above' in places where the 'layer above' exists
    :param img_in:
    :param img_layer:
    :return:
    """

    a_s, a_b, c_s, c_b = prep_operators(source, destination, opacity, offsets)

    co = (a_s * c_s * a_b) + (a_b * c_b * (1 - a_s))
    ao = (a_s * a_b) + (a_b * (1 - a_s))

    return prep_output(co, ao)

def plus(source, destination, opacity=1.0, offsets=(0, 0)):
    """

    :param img_in:
    :param img_layer:
    :return:
    """

    a_s, a_b, c_s, c_b = prep_operators(source, destination, opacity, offsets)

    # np.minimum needed to prevent rollover when max alpha / color values reached
    co = np.minimum((a_s * c_s) + (a_b * c_b), 1.0)
    ao = np.minimum(a_s + a_b, 1.0)

    return prep_output(co, ao)
