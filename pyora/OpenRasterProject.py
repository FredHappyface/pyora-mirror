import sys
import io
import math
import zipfile
import getpass
from PIL import Image
import struct
import os
import xml.etree.cElementTree as ET
from io import BytesIO
from pyora.Render import Renderer, make_thumbnail
from pyora.Layer import Layer, Group
from pyora import TYPE_GROUP, TYPE_LAYER, ORA_VERSION
import re
import uuid

class Project:

    def __init__(self):
        self._children = []
        self._children_paths = {}
        self._children_elems = {}
        self._children_uuids = {}
        self._extracted_merged_image = None
        self._filename_counter = 0
        self._generated_uuids = False


    def __iter__(self):
        for layer in reversed(self._elem_root.findall('.//layer')):
            yield self._children_elems[layer]

    @property
    def iter_layers(self):
        return self.__iter__()

    @property
    def layers_ordered(self):
        return self.__iter__()

    @property
    def groups_ordered(self):
        for group in reversed(self._elem_root.findall('.//stack')):
            if group == self._root_group._elem:
                yield self._root_group
            else:
                yield self._children_elems[group]

    @property
    def layers_and_groups_ordered(self):
        for group in self.groups_ordered:
            yield group
            for layer in reversed(group._elem.findall('layer')):
                yield self._children_elems[layer]

    def _zip_store_image(self, zipref, path, image):
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        imgByteArr.seek(0)
        zipref.writestr(path, imgByteArr.read())

    @staticmethod
    def extract_layer(path_or_file, path=None, uuid=None, pil=False):
        """
        Efficiently extract just one specific layer image
        :param path_or_file: Path to ORA file or file handle
        :param path: Path of layer to extract in the ORA file
        :param uuid: uuid of layer to search for in the ORA file (if path not provided)
        :param pil: for consistency, if true, wrap the image with PIL and return Image()
        otherwise return raw bytes
        :return: bytes or PIL Image()
        """
        with zipfile.ZipFile(path_or_file, 'r') as zipref:
            with zipref.open('stack.xml') as metafile:
                _elem_root = ET.fromstring(metafile.read()).find('stack')
                if path:
                    if path[0] == '/':
                        path = path[1:]
                    for path_part in path.split('/'):
                        _elem_root = _elem_root.find(f"*[@name='{path_part}']")
                        if _elem_root is None:
                            raise ValueError("While following path, part %s not found in ORA!" % path_part)
                else:
                    _elem_root = _elem_root.find(f".//layer[@uuid='{uuid}']")
                    if _elem_root is None:
                        raise ValueError("Unable to find layer with uuid %s in ORA!" % uuid)

            with zipref.open(_elem_root.attrib['src']) as imgdata:
                if pil:
                    return Image.open(imgdata)
                return imgdata.read()

    @staticmethod
    def extract_composite(path_or_file, pil=False):
        """
        Efficiently extract just the composite image
        :param path_or_file: Path to ORA file or file handle
        :param pil: for consistency, if true, wrap the image with PIL and return Image()
        otherwise return raw bytes
        :return: bytes or PIL Image()
        """
        with zipfile.ZipFile(path_or_file, 'r') as zipref:
            with zipref.open('mergedimage.png') as imgdata:
                if pil:
                    return Image.open(imgdata)
                return imgdata.read()

    @staticmethod
    def extract_thumbnail(path_or_file, pil=False):
        """
        Efficiently extract just the thumbnail image
        :param path_or_file: Path to ORA file or file handle
        :param pil: for consistency, if true, wrap the image with PIL and return Image()
        otherwise return raw bytes
        :return: bytes or PIL Image()
        """
        with zipfile.ZipFile(path_or_file, 'r') as zipref:
            with zipref.open('Thumbnails/thumbnail.png') as imgdata:
                if pil:
                    return Image.open(imgdata)
                return imgdata.read()

    @staticmethod
    def load(path_or_file):
        """
        Factory function. Get a new project with data from an existing ORA file
        :param path: path to ORA file to load
        :return: None
        """
        proj = Project()
        proj._load(path_or_file)
        return proj

    def _load(self, path_or_file):

        with zipfile.ZipFile(path_or_file, 'r') as zipref:

            self._children = []
            self._children_paths = {}
            self._children_elems = {}
            self._children_uuids = {}

            # super().__init__(zipref, self)
            with zipref.open('mergedimage.png') as mergedimage:
                self._extracted_merged_image = Image.open(mergedimage)

            try:
                with zipref.open('stack.xml') as metafile:
                    self._elem_root = ET.fromstring(metafile.read())
            except:
                raise ValueError("stack.xml not found in ORA file or not parsable")

            self._elem = self._elem_root[0]  # get the "root" layer group

            def _build_tree(parent, basepath):

                for child_elem in parent._elem:
                    if not child_elem.attrib.get('uuid', None):
                        self._generated_uuids = True
                        child_elem.set('uuid', str(uuid.uuid4()))

                    cur_path = basepath + '/' + child_elem.attrib['name']
                    if child_elem.tag == 'stack':
                        _new = Group(self, child_elem, cur_path)
                        _build_tree(_new, cur_path)
                    elif child_elem.tag == 'layer':
                        with zipref.open(child_elem.attrib['src']) as layerFile:
                            image = Image.open(layerFile).convert('RGBA')
                        _new = Layer(image, self, child_elem, cur_path)
                    else:
                        print(f"Warning: unknown tag in stack: {child_elem.tag}")
                        continue

                    self._children.append(_new)

                    self._children_paths[cur_path] = _new
                    self._children_elems[child_elem] = _new
                    self._children_uuids[_new.uuid] = _new

            self._root_group = Group(self, self._elem, '/')
            _build_tree(self._root_group, '')


    @staticmethod
    def new(width, height, xres=72, yres=72):
        """
        Factory function. Initialize and return a new project.
        :param width: initial width of canvas
        :param height: initial height of canvas
        :param xres: nominal resolution pixels per inch in x
        :param yres: nominal resolution pixels per inch in y
        :return: None
        """
        proj = Project()
        proj._new(width, height, xres, yres)
        return proj

    def _new(self, width, height, xres, yres):

        self._elem_root = ET.fromstring(f'<image version="{ORA_VERSION}" h="{height}" w="{width}" '
                                        f'xres="{xres}" yres="{yres}">'
                                        f'<stack composite-op="svg:src-over" opacity="1" name="root" '
                                        f'visibility="visible"></stack></image>')
        self._elem = self._elem_root[0]
        self._root_group = Group(self, self._elem, '/')
        self._extracted_merged_image = None

    def save(self, path_or_file, composite_image=None, use_original=False):
        """
        Save the current project state to an ORA file.
        :param path: path to the ora file to save
        :param composite_image: - PIL Image() object of the composite rendered canvas. It is used to create the
        mergedimage full rendered preview, as well as the thumbnail image. If not provided, we will attempt to
        generate one by stacking all of the layers in the project. Note that the image you pass may be modified
        during this process, so if you need to use it elsewhere in your code, you should copy() first.
        :param use_original: IF true, and If there was a stored 'mergedimage' already in the file which was opened,
        use that for the 'mergedimage' in the new file
        :return: None
        """
        with zipfile.ZipFile(path_or_file, 'w') as zipref:

            zipref.writestr('mimetype', "image/openraster".encode())
            zipref.writestr('stack.xml', ET.tostring(self._elem_root, method='xml'))

            if not composite_image:
                if use_original and self._extracted_merged_image:
                    composite_image = self._extracted_merged_image
                else:
                    # render using our built in library
                    r = Renderer(self)
                    composite_image = r.render()
            self._zip_store_image(zipref, 'mergedimage.png', composite_image)

            make_thumbnail(composite_image)  # works in place
            self._zip_store_image(zipref, 'Thumbnails/thumbnail.png', composite_image)

            for layer in self.children:
                if layer.type == TYPE_LAYER:
                    self._zip_store_image(zipref, layer['src'], layer.get_image_data())



    def _get_parent_from_path(self, path):
        parent_path = '/'.join(path.split('/')[:-1])
        if not parent_path:
            return self._root_group
        return self._children_paths[parent_path]

    def _split_path_index(self, path):
        """
        Get tuple of (path, index) from indexed path
        """
        found = re.findall(r'(.*)\[(\d+)\]', path)
        return found[0] if found else (path, 1)

    def _add_elem(self, tag, path, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True, composite_op="svg:src-over",
                  **kwargs):

        print(kwargs)
        if not 'uuid' in kwargs or kwargs['uuid'] is None:
            self._generated_uuids = True
            kwargs['uuid'] = str(uuid.uuid4())

        parts = path.split('/')
        name, parent_elem = parts[-1], self._get_parent_from_path(path)._elem
        new_elem = ET.Element(tag, {'name': name, 'x': str(offsets[0]), 'y': str(offsets[1]),
                                        'visibility': 'visible' if visible else 'hidden',
                                        'opacity': str(opacity), 'composite-op': composite_op,
                                    **{k: str(v) for k, v in kwargs.items() if v is not None}})
        parent_elem.insert(z_index - 1, new_elem)
        return new_elem

    def _add_layer(self, image, path, **kwargs):
        # generate some unique filename
        # we follow Krita's standard of just 'layer%d' type format
        #index = len([x for x in self.children if x.type == TYPE_LAYER])
        new_filename = f'/data/layer{self._filename_counter}.png'
        self._filename_counter += 1;

        # add xml element
        elem = self._add_elem('layer', path, **kwargs, src=new_filename)
        obj = Layer(image, self, elem, path)

        self.children.append(obj)
        self._children_paths[path] = obj
        self._children_elems[elem] = obj
        self._children_uuids[obj.uuid] = obj

        return obj

    # def delete_path(self, path):
    #     item = self._children_paths[path]
    #     del self._children_paths[path]
    #     del self._children_elems[item._elem]
    #     if item.uuid:
    #         del self._children_uuids[item.uuid]
    #     self.children.remove(item)

    def _add_group(self, path, **kwargs):
        elem = self._add_elem('stack', path, **kwargs)
        obj = Group(self, elem, path)

        if not 'isolation' in kwargs:
            kwargs['isolation'] = 'isolate'

        self.children.append(obj)
        self._children_paths[path] = obj
        self._children_elems[elem] = obj
        self._children_uuids[obj.uuid] = obj
        return obj

    def _make_groups_recursively(self, path, as_group=True):
        """
        Create groups, kind of like $ mkdir -p
        :param path: absolute path
        :param as_group: if true, assume abspath(path) should be a layer, and don't make it. If False, make the
        whole path.
        :return:
        """

        # absolute path slash is for styling/consistency only, remove it if exists
        if path[0] == '/':
            path = path[1:]

        # determine if the required group exists yet
        # and add all required groups to make the needed path
        parts = path.split('/')
        parent_path = '/'.join(parts[:-(1 if as_group else 0)])

        if not parent_path in self._children_paths:
            for i, _parent_name in enumerate(parts[(1 if as_group else 0):], 1):
                _sub_parent_path = '/' + '/'.join(parts[:i])
                if not _sub_parent_path in self._children_paths:
                    # make new empty group
                    self._add_group(_sub_parent_path, isolation='isolate')

    def add_layer(self, image, path=None, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", uuid=None, **kwargs):
        """
        Append a new layer to the project
        :param image: a PIL Image() object containing the image data to add
        :param path: Absolute filesystem-like path of the layer in the project. For example "/layer1" or
        "/group1/layer2". If given without a leading slash, like "layer3", we assume the layer is placed at
        the root of the project. If omitted or set to None, path is set to the filename of the input image.
        :param offsets: tuple of (x, y) offset from the top-left corner of the Canvas
        :param opacity: float - layer opacity 0.0 to 1.0
        :param visible: bool - is the layer visible
        :param composite_op: str - composite operation attribute passed directly to stack / layer element
        :return: Layer() - reference to the newly created layer object
        """
        if path is None or not path:
            path = image.filename.split('/')[-1]

        self._make_groups_recursively(path)

        if not path[0] == '/':
            path = '/' + path

        # make the new layer itself
        return self._add_layer(image, path, z_index=z_index, offsets=offsets, opacity=opacity, visible=visible,
                        composite_op=composite_op, uuid=uuid, **kwargs)

    def add_group(self, path, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", uuid=None, isolated=True, **kwargs):
        """
        Append a new layer group to the project
        :param path: Absolute filesystem-like path of the group in the project. For example "/group1" or
        "/group1/group2". If given without a leading slash, like "group3", we assume the group is placed at
        the root of the project.
        :param offsets: tuple of (x, y) offset from the top-left corner of the Canvas
        :param opacity: float - group opacity 0.0 to 1.0
        :param visible: bool - is the group visible
        :param composite_op: str - composite operation attribute passed directly to stack / layer element
        :param uuid: str - uuid identifier value for this group
        :param isolation:bool - True or False
        :return: Layer() - reference to the newly created layer object
        """
        self._make_groups_recursively(path)

        if not path[0] == '/':
            path = '/' + path

        kwargs['isolation'] = 'isolate' if isolated else 'auto'

        # make the new group itself
        return self._add_group(path, z_index=z_index, offsets=offsets, opacity=opacity, visible=visible,
                        composite_op=composite_op, uuid=uuid, **kwargs)

    def remove(self, path=None, uuid=None):
        """
        Remove some layer or group and all of its children from the project
        :param path:
        :param uuid:
        :return:
        """
        if not path[0] == '/':
            path = '/' + path

        if path:
            root_child = self[path]
        else:
            root_child = self.get_by_uuid(uuid)

        # this removes all of the python references
        children_to_remove = []
        for _path in self._children_paths:
            if _path == root_child.path:
                # remove XML elementtree reference
                parent_elem = self._get_parent_from_path(_path)._elem
                parent_elem.remove(self._children_paths[_path]._elem)
            if _path.startswith(root_child.path):
                children_to_remove.append(self._children_paths[_path])

        for child in children_to_remove:
            del self._children_elems[child._elem]
            del self._children_paths[child.path]
            if child.uuid:
                del self._children_uuids[child.uuid]
            self._children.remove(child)

    def move(self, path=None, uuid=None, dest_path=None, dest_uuid=None, dest_z_index=1):
        """
        Move some layer or group and all of its children somewhere else inside the project
        If there are some layer groups that are missing for the destination to exist, they
        will be created automatically.
        :param path: source group/layer path to move
        :param uuid: source group/layer uuid to move
        :param dest_path: dest group path to place source element inside of
        :param dest_path: dest group uuid to place source element inside of
        :param dest_z_index: inside of the destination group, place the moved layer/group at this index
        :return: None
        """

        # just find all the individual elements below the selected one by path and add them all
        # iteratively. Less efficient for now but I don't think anyone is keeping track.

        # or: better solution: just find all of the paths that are below, and change their project
        # child_paths entry with the path replacement and

        if not path[0] == '/':
            path = '/' + path

        if path:
            root_child = self[path]
        else:
            root_child = self.get_by_uuid(uuid)

        if dest_uuid:
            dest_path = self.get_by_uuid(uuid).path

        self._make_groups_recursively(dest_path, as_group=False)

        move_paths = []
        for _path in self._children_paths:
            if _path == root_child.path:
                # move element in the XML object repr
                old_parent_elem = self._get_parent_from_path(_path)._elem
                new_parent_elem = self._children_paths[dest_path]._elem
                old_parent_elem.remove(root_child._elem)
                new_parent_elem.insert(dest_z_index-1, root_child._elem)

            if _path.startswith(root_child.path):
                move_paths.append(_path)

        for _path in move_paths:

            full_dest_path = _path.replace(root_child.path, dest_path + ('' if dest_path[-1] == '/' else '/')  + root_child.name, 1)
            self._children_paths[full_dest_path] = self._children_paths.pop(_path)
            self._children_paths[full_dest_path]._path = full_dest_path

    @property
    def dimensions(self):
        """
        Project (width, height) dimensions in px
        :return: (width, height) tuple
        """
        return int(self._elem_root.attrib['w']), int(self._elem_root.attrib['h'])

    @property
    def ppi(self):
        if 'xres' in self._elem_root.attrib and 'yres' in self._elem_root.attrib:
            return self._elem_root.attrib['xres'], self._elem_root.attrib['yres']
        else:
            return None

    @property
    def name(self):
        return self._elem_root.attrib.get('name', None)

    @property
    def children(self):
        return self._children

    @property
    def paths(self):
        return self._children_paths

    @property
    def uuids(self):
        return self._children_uuids

    @property
    def root(self):
        """
        Get a reference to the outermost layer group containing everything else
        :return: Group() Object
        """
        return self._root_group

    def __contains__(self, item):
        return item in self._children_paths

    def __getitem__(self, item):
        return self._children_paths[item]

    def get_by_uuid(self, uuid):
        return self._children_uuids[uuid]

    def get_image_data(self, use_original=False):
        """
        Get a PIL Image() object of the entire project (composite)
        :param use_original: IF true, and If there was a stored 'mergedimage' already in the file which was opened,
        just return that. In any other case a new merged image is generated.
        :return: PIL Image()
        """

        if self._extracted_merged_image and use_original:
            return self._extracted_merged_image

        r = Renderer(self)
        return r.render()

    def get_thumbnail_image_data(self, use_original=False):
        """
        Get a PIL Image() object of the entire project (composite) (standard 256x256 max ORA thumbnail size
        :param use_original: IF true, and If there was a stored 'mergedimage' already in the file which was opened,
        just return that. In any other case a new merged image is generated.
        :return: PIL Image()
        """
        if self._extracted_merged_image and use_original:
            return make_thumbnail(self._extracted_merged_image)

        r = Renderer(self)
        return make_thumbnail(r.render())






