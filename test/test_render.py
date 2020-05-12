from pyora import Project, Renderer, TYPE_LAYER
from PIL import Image
import os, sys
from utils import getf

def test_basic_render(tmp_path):

    save_to = str(tmp_path.joinpath('test.png'))

    proj = Project.load(getf('stacked-group-passthrough-alpha.ora'))
    rend = Renderer(proj)

    rend.render().save(save_to)

    img = Image.open(save_to)

    assert img.size == (100, 117)

