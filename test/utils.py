from pyora import Project, Renderer, TYPE_LAYER
from PIL import Image
import os, sys

base_test_path = os.path.join(os.path.dirname(__file__), 'test_files')
def getf(name):
    return os.path.join(base_test_path, name)