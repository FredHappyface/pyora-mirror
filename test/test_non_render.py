from pyora import Project, Renderer, TYPE_LAYER
from PIL import Image
import os, sys
from utils import getf

def test_load_and_save(tmp_path):
    save_to = str(tmp_path.joinpath('tmp_project_out.ora'))

    proj = Project.load(getf('stacked-group-passthrough-alpha.ora'))

    proj.save(save_to)

    # make sure it can still be loaded
    proj.load(save_to)

def test_create_from_new(tmp_path):
    save_to = str(tmp_path.joinpath('tmp_project_out.ora'))

    proj = Project.new(100, 100)

    proj.add_layer(Image.open(getf('1.jpg')))

    assert len(proj.children) == 1
    assert len(proj.children_recursive) == 1

    proj.save(save_to)

    # make sure it can still be loaded
    proj.load(save_to)

    assert len(proj.children) == 1
    assert len(proj.children_recursive) == 1

def test_move_layer(tmp_path):
    proj = Project.new(100, 100)

    layer1 = proj.add_layer(Image.open(getf('1.jpg')), 'layer 1')
    layer2 = proj.add_layer(Image.open(getf('1.jpg')), 'layer 2')

    group1 = proj.add_group('group1')
    proj.move(layer2.uuid, group1.uuid)

    assert list(group1.children)[0].name == 'layer 2'
