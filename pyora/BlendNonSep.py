import numpy as np
from pyora.Blend import reshape_dest, _compose_alpha

"""
Implementation of the non-separable blending modes as described in 

https://www.w3.org/TR/compositing-1/#blendingnonseparable

"""


"""
four non-separable utility functions as described on the aforementioned page

Lum(C) = 0.3 x Cred + 0.59 x Cgreen + 0.11 x Cblue
    
    ClipColor(C)
        L = Lum(C)
        n = min(Cred, Cgreen, Cblue)
        x = max(Cred, Cgreen, Cblue)
        if(n < 0)
            C = L + (((C - L) * L) / (L - n))
                      
        if(x > 1)
            C = L + (((C - L) * (1 - L)) / (x - L))
        
        return C
    
    SetLum(C, l)
        d = l - Lum(C)
        Cred = Cred + d
        Cgreen = Cgreen + d
        Cblue = Cblue + d
        return ClipColor(C)
        
    Sat(C) = max(Cred, Cgreen, Cblue) - min(Cred, Cgreen, Cblue)

"""

def _lum(_c):
    """

    :param c: x by x by 3 matrix of rgb color components of pixels
    :return: x by x by 3 matrix of luminosity of pixels
    """

    return (_c[:, :, 0] * 0.299) + (_c[:, :, 1] * 0.587) + (_c[:, :, 2] * 0.114)

def _setLum(c_orig, l):
    _c = c_orig.copy()
    _l = _lum(_c)
    d = l - _l
    _c[:, :, 0] += d
    _c[:, :, 1] += d
    _c[:, :, 2] += d
    _l = _lum(_c)

    _n = np.min(_c, axis=2)
    _x = np.max(_c, axis=2)

    for i in range(_c.shape[0]):
        for j in range(_c.shape[1]):
            c = _c[i][j]
            l = _l[i, j]
            n = _n[i, j]
            x = _x[i, j]

            if n < 0:
                _c[i][j] = l + (((c - l) * l) / (l - n))

            if x > 1:
                _c[i][j] = l + (((c - l) * (1 - l)) / (x - l))

    return _c

def _sat(_c):
    """

    :param c: x by x by 3 matrix of rgb color components of pixels
    :return: int of saturation of pixels
    """
    return np.max(_c, axis=2) - np.min(_c, axis=2)

# def _setSatKern(c):
#     max_i = np.argmax(c)
#     min_i = np.argmin(c)
#     if max_i != 2 and min_i != 2:
#         mid_i = 2
#     elif max_i != 1 and min_i != 1:
#         mid_i = 1
#     else:
#         mid_i = 0
#
#     if c[max_i] > c[min_i]:
#         c[mid_i] = (((c[mid_i] - c[min_i]) * s) / (c[max_i] - c[min_i]))
#         c[max_i] = s
#     else:
#         c[mid_i] = 0
#         c[max_i] = 0
#     c[min_i] = 0
#     return c

#setSatKern = np.vectorize(_setSatKern)

def _setSat(c_orig, s):
    """
    Set a new saturation value for the matrix of color

    The current implementation cannot be vectorized in an efficient manner, so it is very slow,
    O(m*n) at least. This might be able to be improved with openCL if that is the direction that the lib takes.
    :param c: x by x by 3 matrix of rgb color components of pixels
    :param s: int of the new saturation value for the matrix
    :return: x by x by 3 matrix of luminosity of pixels
    """
    _c = c_orig.copy()


    for i in range(_c.shape[0]):
        for j in range(_c.shape[1]):
            c = _c[i][j]

            min_i = 0
            mid_i = 1
            max_i = 2

            # this part might be able to be a kernel in the future
            if c[mid_i] < c[min_i]:
                tmp = min_i
                min_i = mid_i
                mid_i = tmp

            if c[max_i] < c[mid_i]:
                tmp = mid_i
                mid_i = max_i
                max_i = tmp

            if c[mid_i] < c[min_i]:
                tmp = min_i
                min_i = mid_i
                mid_i = tmp

            if c[max_i] - c[min_i] > 0.0:
                _c[i][j][mid_i] = (((c[mid_i] - c[min_i]) * s[i, j]) / (c[max_i] - c[min_i]))
                _c[i][j][max_i] = s[i, j]
            else:
                _c[i][j][mid_i] = 0
                _c[i][j][max_i] = 0
            _c[i][j][min_i] = 0

    return _c


def _general_blend(source, destination, opacity, offsets, blend_func):
    source = reshape_dest(source, destination, offsets)

    destination_norm = destination / 255.0
    source_norm = source / 255.0

    Cb = destination_norm[:, :, :3]
    Cs = source_norm[:, :, :3]

    comp = blend_func(Cs, Cb)

    ratio = _compose_alpha(destination_norm, source_norm, opacity)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + source_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, source_norm[:, :, 3])))  # add alpha channel and replace nans

    return img_out * 255.0

import math

def _colourCompositingFormula(_as, ab, ar, Cs, Cb, Bbs):
    return (1 - (_as / ar)) * Cb + (_as / ar) * math.floor((1 - ab) * Cs + ab * Bbs)


def hue(source, destination, opacity, offsets=(0, 0)):
    """

    Creates a color with the hue of the source color and the saturation and luminosity of the backdrop color.

    """

    def _blend_func(Cs, Cb):
        return _setLum(_setSat(Cs, _sat(Cb)), _lum(Cb))

    return _general_blend(source, destination, opacity, offsets, _blend_func)


def saturation(source, destination, opacity, offsets=(0, 0)):
    """

    Creates a color with the saturation of the source color and the hue and luminosity of the backdrop color.

    """

    def _blend_func(Cs, Cb):
        return _setLum(_setSat(Cb, _sat(Cs)), _lum(Cb))

    return _general_blend(source, destination, opacity, offsets, _blend_func)

def color(source, destination, opacity, offsets=(0, 0)):
    """

    Creates a color with the hue and saturation of the source color and the luminosity of the backdrop color.

    """

    def _blend_func(Cs, Cb):
        return _setLum(Cs, _lum(Cb))

    return _general_blend(source, destination, opacity, offsets, _blend_func)

def luminosity(source, destination, opacity, offsets=(0, 0)):
    """

    Creates a color with the luminosity of the source color and the hue and saturation of the backdrop color.

    """

    def _blend_func(Cs, Cb):
        return _setLum(Cb, _lum(Cs))

    return _general_blend(source, destination, opacity, offsets, _blend_func)
