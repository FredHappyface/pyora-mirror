from pyora import Project, Renderer, TYPE_LAYER
from PIL import Image
import os, sys
from utils import getf
import numpy as np

def test_basic_render(tmp_path):

    save_to = str(tmp_path.joinpath('test.png'))

    proj = Project.load(getf('stacked-group-passthrough-alpha.ora'))
    rend = Renderer(proj)

    rend.render().save(save_to)

    img = Image.open(save_to)

    assert img.size == (100, 117)

def test_add_layer_ordering(tmp_path):

    save_to = str(tmp_path.joinpath('test.png'))

    red_pixel = Image.fromarray(np.array([[[255, 0, 0]]], dtype=np.uint8), 'RGB')
    green_pixel = Image.fromarray(np.array([[[0, 255, 0]]], dtype=np.uint8), 'RGB')
    blue_pixel = Image.fromarray(np.array([[[0, 0, 255]]], dtype=np.uint8), 'RGB')

    proj = Project.new(1, 1)
    proj.add_layer(red_pixel, 'red')
    proj.add_layer(green_pixel, 'green')
    proj.add_layer(blue_pixel, 'blue')

    rend = Renderer(proj)
    rend.render().save(save_to)
    img = Image.open(save_to)

    # by default, layers are stacked downward
    assert list(np.asarray(img)[0][0]) == [255, 0, 0, 255]

    # this can be overridden by using explicit z index
    proj = Project.new(1, 1)
    proj.add_layer(red_pixel, 'red', z_index=1)
    proj.add_layer(green_pixel, 'green', z_index=2)
    proj.add_layer(blue_pixel, 'blue', z_index=3)

    rend = Renderer(proj)
    rend.render().save(save_to)
    img = Image.open(save_to)

    assert list(np.asarray(img)[0][0]) == [0, 0, 255, 255]

