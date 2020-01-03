"""
.. module:: blend_modes

This page documents all function available in blend_modes in detail. If this documentation cannot answer your questions,
please raise an issue on `blend_modes' GitHub page <https://github.com/flrs/blend_modes/issues>`__.

Overview
--------

.. currentmodule:: blend_modes.blending_functions

.. autosummary::
    :nosignatures:

    addition
    darken_only
    difference
    divide
    dodge
    grain_extract
    grain_merge
    hard_light
    lighten_only
    multiply
    normal
    overlay
    screen
    soft_light
    subtract


Note:
    All examples on this page are blends of two images: As a bottom layer, there is a rainbow-filled square with
    some transparent border on the right and bottom edges. The top layer is a small rectangle that is
    filled with a colorful circular gradient. The top layer is blended upon the bottom layer with 50%
    transparency in all of the examples below.

    .. |logo1| image:: ../tests/orig.png
        :scale: 30%

    .. |logo2| image:: ../tests/layer.png
        :scale: 30%


    .. table:: Bottom and top layers for blending examples
       :align: center

       +---------+---------+
       | |logo1| | |logo2| |
       +---------+---------+

Detailed Documentation
----------------------

"""
import numpy as np



def _compose_alpha(img_layer, img_in, opacity, window=None):
    """Calculate alpha composition ratio between two images.
    window: tuple of min_x, max_x, min_y, max_y to actually blend
    """

    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.NAN] = 0.0

    # make sure to get a full mask on parts of the image which are not part of both source and backdrop
    if window:
        mask = np.ones_like(ratio, dtype=bool)
        mask[window[0]:window[1], window[2]:window[3]] = False
        ratio[mask] = 1.0
    return ratio

def reshape_img_in(img_layer, img_in, offsets):
    # shift img_in by offset if needed by adding rows and cols of zeros before

    if offsets[0] > 0:
        img_layer = np.hstack((np.zeros((img_layer.shape[0], offsets[0], 4), dtype=np.float64), img_layer))
    elif offsets[0] < 0:
        if offsets[0] > -1*img_layer.shape[1]:
            img_layer = img_layer[:, -1 * offsets[0]:, :]
        else:
            # offset offscreen completely, there is nothing left
            return np.zeros(img_in.shape, dtype=np.float64)
    if offsets[1] > 0:
        img_layer = np.vstack((np.zeros((offsets[1], img_layer.shape[1], 4), dtype=np.float64), img_layer))
    elif offsets[1] < 0:
        if offsets[1] > -1 * img_layer.shape[0]:
            img_layer = img_layer[-1 * offsets[1]:, :, :]
        else:
            # offset offscreen completely, there is nothing left
            return np.zeros(img_in.shape, dtype=np.float64)


    # resize array to fill small images with zeros
    if img_layer.shape[0] < img_in.shape[0]:
        img_layer = np.vstack(
            (img_layer, np.zeros((img_in.shape[0] - img_layer.shape[0], img_layer.shape[1], 4), dtype=np.float64)))
    if img_layer.shape[1] < img_in.shape[1]:
        img_layer = np.hstack(
            (img_layer, np.zeros((img_layer.shape[0], img_in.shape[1] - img_layer.shape[1], 4), dtype=np.float64)))

    # crop the source if the backdrop is smaller
    img_layer = img_layer[:img_in.shape[0], :img_in.shape[1], :]

    return img_layer


def normal(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply "normal" blending mode of a layer on an image.

    Example:
        .. image:: ../tests/normal_50p.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import normal
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = normal(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on `Wikipedia <https://en.wikipedia.org/wiki/Alpha_compositing#Description>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image
    """

    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    # Add alpha-channels, if they are not provided
    # if img_in_norm.shape[2] == 3:
    #     img_in_norm = np.dstack((img_in_norm, np.zeros(img_in_norm.shape[:2] + (3,))))
    # if img_layer_norm.shape[2] == 3:
    #     img_layer_norm = np.dstack((img_layer_norm, np.zeros(img_layer_norm.shape[:2] + (3,))))

    # Extract alpha-channels and apply opacity
    img_in_alp = np.expand_dims(img_in_norm[:, :, 3], 2)  # alpha of b, prepared for broadcasting
    img_layer_alp = np.expand_dims(img_layer_norm[:, :, 3], 2) * opacity  # alpha of a, prepared for broadcasting

    # Blend images

    with np.errstate(invalid='ignore'):
        c_out = (img_layer_norm[:, :, :3] * img_layer_alp + img_in_norm[:, :, :3] * img_in_alp * (1 - img_layer_alp)) \
            / (img_layer_alp + img_in_alp * (1 - img_layer_alp))

    # Blend alpha
    cout_alp = img_layer_alp + img_in_alp * (1 - img_layer_alp)

    # Combine image and alpha
    c_out = np.dstack((c_out, cout_alp))

    np.nan_to_num(c_out, copy=False)

    return c_out * 255.0


def soft_light(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply soft light blending mode of a layer on an image.

    Example:
        .. image:: ../tests/soft_light.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import soft_light
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = soft_light(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Soft_Light>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    # The following code does this:
    #   multiply = img_in_norm[:, :, :3]*img_layer[:, :, :3]
    #   screen = 1.0 - (1.0-img_in_norm[:, :, :3])*(1.0-img_layer[:, :, :3])
    #   comp = (1.0 - img_in_norm[:, :, :3]) * multiply + img_in_norm[:, :, :3] * screen
    #   ratio_rs = np.reshape(np.repeat(ratio,3),comp.shape)
    #   img_out = comp*ratio_rs + img_in_norm[:, :, :3] * (1.0-ratio_rs)

    comp = (1.0 - img_in_norm[:, :, :3]) * img_in_norm[:, :, :3] * img_layer_norm[:, :, :3] \
           + img_in_norm[:, :, :3] * (1.0 - (1.0 - img_in_norm[:, :, :3]) * (1.0 - img_layer_norm[:, :, :3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def lighten_only(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply lighten only blending mode of a layer on an image.

    Example:
        .. image:: ../tests/lighten_only.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import lighten_only
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = lighten_only(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Lighten_Only>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.maximum(img_in_norm[:, :, :3], img_layer_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def screen(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply screen blending mode of a layer on an image.

    Example:
        .. image:: ../tests/screen.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import screen
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = screen(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Screen>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = 1.0 - (1.0 - img_in_norm[:, :, :3]) * (1.0 - img_layer_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def dodge(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply dodge blending mode of a layer on an image.

    divides the bottom layer by the inverted top layer

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.minimum(img_in_norm[:, :, :3] / ((1.0 + np.finfo(np.float64).eps) - img_layer_norm[:, :, :3]), 1.0)


    ratio_rs = np.nan_to_num(np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]]))

    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans

    return img_out * 255.0

def burn(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply burn blending mode of a layer on an image.

    divides the inverted bottom layer by the top layer, and then inverts the result

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)


    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0


    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    # in : botton  , layer: top
    comp = np.maximum(1.0 - (((1.0 + np.finfo(np.float64).eps) - img_in_norm[:, :, :3]) / img_layer_norm[:, :, :3]), 0.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans

    return img_out * 255.0

def addition(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply addition blending mode of a layer on an image.

    Example:
        .. image:: ../tests/addition.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import addition
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = addition(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Addition>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = img_in_norm[:, :, :3] + img_layer_norm[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def darken_only(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply darken only blending mode of a layer on an image.

    Example:
        .. image:: ../tests/darken_only.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import darken_only
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = darken_only(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Darken_Only>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.minimum(img_in_norm[:, :, :3], img_layer_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def multiply(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply multiply blending mode of a layer on an image.

    Example:
        .. image:: ../tests/multiply.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import multiply
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = multiply(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Multiply>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_layer_norm[:, :, :3] * img_in_norm[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def hard_light(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply hard light blending mode of a layer on an image.

    Example:
        .. image:: ../tests/hard_light.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import hard_light
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = hard_light(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Hard_Light>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.greater(img_layer_norm[:, :, :3], 0.5) \
           * np.minimum(1.0 - ((1.0 - img_in_norm[:, :, :3])
                               * (1.0 - (img_layer_norm[:, :, :3] - 0.5) * 2.0)), 1.0) \
           + np.logical_not(np.greater(img_layer_norm[:, :, :3], 0.5)) \
           * np.minimum(img_in_norm[:, :, :3] * (img_layer_norm[:, :, :3] * 2.0), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def difference(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply difference blending mode of a layer on an image.

    Example:
        .. image:: ../tests/difference.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import difference
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = difference(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Difference>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = img_in_norm[:, :, :3] - img_layer_norm[:, :, :3]
    comp[comp < 0.0] *= -1.0

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def subtract(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply subtract blending mode of a layer on an image.

    Example:
        .. image:: ../tests/subtract.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import subtract
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = subtract(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Subtract>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = img_in[:, :, :3] - img_layer_norm[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def grain_extract(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply grain extract blending mode of a layer on an image.

    Example:
        .. image:: ../tests/grain_extract.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import grain_extract
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = grain_extract(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information in the `GIMP Documentation <https://docs.gimp.org/en/gimp-concepts-layer-modes.html>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_in_norm[:, :, :3] - img_layer_norm[:, :, :3] + 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def grain_merge(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply grain merge blending mode of a layer on an image.

    Example:
        .. image:: ../tests/grain_merge.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import grain_merge
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = grain_merge(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information in the `GIMP Documentation <https://docs.gimp.org/en/gimp-concepts-layer-modes.html>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_in_norm[:, :, :3] + img_layer_norm[:, :, :3] - 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def divide(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply divide blending mode of a layer on an image.

    Example:
        .. image:: ../tests/divide.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import divide
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = divide(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Divide>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.minimum((256.0 / 255.0 * img_in_norm[:, :, :3]) / (1.0 / 255.0 + img_layer_norm[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def overlay(img_layer, img_in, opacity, offsets=(0, 0)):
    """Apply overlay blending mode of a layer on an image.

    Note:
        The implementation of this method was changed in version 2.0.0. Previously, it would be identical to the
        soft light blending mode. Now, it resembles the implementation on Wikipedia. You can still use the soft light
        blending mode if you are looking for backwards compatibility.

    Example:
        .. image:: ../tests/overlay.png
            :width: 30%

        ::

            import cv2, numpy
            from blend_modes import overlay
            img_in = cv2.imread('./orig.png', -1).astype(float)
            img_layer = cv2.imread('./layer.png', -1).astype(float)
            img_out = overlay(img_in,img_layer,0.5)
            cv2.imshow('window', img_out.astype(numpy.uint8))
            cv2.waitKey()

    See Also:
        Find more information on
        `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=868545948#Overlay>`__.

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending
      disable_type_checks(bool): Whether type checks within the function should be disabled. Disabling the checks may
        yield a slight performance improvement, but comes at the cost of user experience. If you are certain that
        you are passing in the right arguments, you may set this argument to 'True'. Defaults to 'False'.

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """
    img_layer = reshape_img_in(img_layer, img_in, offsets)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.less(img_in_norm[:, :, :3], 0.5) * (2 * img_in_norm[:, :, :3] * img_layer_norm[:, :, :3]) \
           + np.greater_equal(img_in_norm[:, :, :3], 0.5) \
           * (1 - (2 * (1 - img_in_norm[:, :, :3]) * (1 - img_layer_norm[:, :, :3])))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_layer_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_layer_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0