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
        self._children_elems = {}
        self._children_uuids = {}
        self._extracted_merged_image = None
        self._filename_counter = 0
        self._generated_uuids = False

    def get_by_path(self, path):
        """

        :return:
        """

        if path == '/':
            return self.root
        if path.startswith('/'):
            path = path[1:]

        current_group = self._root_group
        for name in path.split('/'):
            found = False
            for child in current_group.children:
                if child.name == name:
                    current_group = child
                    found = True
                    break

            if not found:
                raise Exception(f"Layer with path {path} was not found")

        return current_group

    def get_by_uuid(self, uuid):
        return self._children_uuids[uuid]

    def _parentNode(self, elem):
        """
        Get the parent node of elem, based on ElementTree Limitations
        """
        uuid = elem.attrib['uuid']
        return self._elem_root.find(f'.//*[@uuid="{uuid}"]...')

    @property
    def children_recursive(self):
        return self._children

    @property
    def children(self):
        children = []
        for _child in self._root_group._elem:
            children.append(self.get_by_uuid(_child.attrib['uuid']))

        return children

    @property
    def root(self):
        """
        Get a reference to the outermost layer group containing everything else
        :return: Group() Object
        """
        return self._root_group

    @property
    def uuids(self):
        return self._children_uuids

    @property
    def iter_layers(self):
        for layer in reversed(self._elem_root.findall('.//layer')):
            yield self._children_elems[layer]

    @property
    def iter_groups(self):
        for group in reversed(self._elem_root.findall('.//stack')):
            if group == self._root_group._elem:
                yield self._root_group
            else:
                yield self._children_elems[group]

    def __iter__(self):
        """
        Same as .children
        :return:
        """
        return self.iter_layers


    def __contains__(self, path):
        try:
            self.get_by_path(path)
        except:
            return False
        return True

    def __getitem__(self, path):
        return self.get_by_path(path)

    @property
    def layers_and_groups_ordered(self):
        for group in self.iter_groups:
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

            # we expect certain default attributes for the root group (Krita follows this standard)
            self._elem.set("isolation", "isolate")
            self._elem.set("composite-op", "svg:src-over")
            self._elem.set("opacity", "1")
            self._elem.set("name", "")
            self._elem.set("visibility", "visible")

            def _build_tree(parent, basepath):

                for child_elem in parent._elem:
                    if not child_elem.attrib.get('uuid', None):
                        self._generated_uuids = True
                        child_elem.set('uuid', str(uuid.uuid4()))

                    cur_path = basepath + '/' + child_elem.attrib['name']
                    if child_elem.tag == 'stack':
                        _new = Group(self, child_elem)
                        _build_tree(_new, cur_path)
                    elif child_elem.tag == 'layer':
                        with zipref.open(child_elem.attrib['src']) as layerFile:
                            image = Image.open(layerFile).convert('RGBA')
                        _new = Layer(image, self, child_elem)
                    else:
                        print(f"Warning: unknown tag in stack: {child_elem.tag}")
                        continue

                    self._children.append(_new)

                    self._children_elems[child_elem] = _new
                    self._children_uuids[_new.uuid] = _new

            self._root_group = Group(self, self._elem)
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
                                        f'visibility="visible" isolation="isolate"></stack></image>')
        self._elem = self._elem_root[0]
        self._root_group = Group(self, self._elem)
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

            for layer in self.children_recursive:
                if layer.type == TYPE_LAYER:
                    self._zip_store_image(zipref, layer['src'], layer.get_image_data())

    def _get_parent_from_path(self, path):

        parent_path = '/'.join(path.split('/')[:-1])

        if parent_path == '':
            return self._root_group
        
        return self.get_by_path(parent_path)

    def _insertElementAtIndex(self, parent, index, element):

        parent.insert(index, element)

    def _split_path_index(self, path):
        """
        Get tuple of (path, index) from indexed path
        """
        found = re.findall(r'(.*)\[(\d+)\]', path)
        return found[0] if found else (path, 1)

    def _add_elem(self, tag, parent_elem, name, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True, composite_op="svg:src-over",
                  **kwargs):

        if not 'uuid' in kwargs or kwargs['uuid'] is None:
            self._generated_uuids = True
            kwargs['uuid'] = str(uuid.uuid4())

        new_elem = ET.Element(tag, {'name': name, 'x': str(offsets[0]), 'y': str(offsets[1]),
                                        'visibility': 'visible' if visible else 'hidden',
                                        'opacity': str(opacity), 'composite-op': composite_op,
                                    **{k: str(v) for k, v in kwargs.items() if v is not None}})
        parent_elem.insert(z_index - 1, new_elem)
        return new_elem

    def _add_layer(self, image, parent_elem, name, **kwargs):
        # generate some unique filename
        # we follow Krita's standard of just 'layer%d' type format
        #index = len([x for x in self.children if x.type == TYPE_LAYER])
        new_filename = f'/data/layer{self._filename_counter}.png'
        self._filename_counter += 1

        # add xml element
        elem = self._add_elem('layer', parent_elem, name, **kwargs, src=new_filename)
        obj = Layer(image, self, elem)

        self._children.append(obj)
        self._children_elems[elem] = obj
        self._children_uuids[obj.uuid] = obj

        return obj

    def _add_group(self, parent_elem, name, **kwargs):
        elem = self._add_elem('stack', parent_elem, name, **kwargs)
        obj = Group(self, elem)

        if not 'isolation' in kwargs:
            kwargs['isolation'] = 'isolate'

        self._children.append(obj)
        self._children_elems[elem] = obj
        self._children_uuids[obj.uuid] = obj
        return obj

    def _make_groups_recursively(self, path):
        """
        creates all of the groups which would be required UNDER the specified path (not the final, deepest path element)
        as this works with paths it will just choose the first matching path if duplicate names are found
        :return:
        """

        # absolute path slash is for styling/consistency only, remove it if exists
        if path[0] == '/':
            path = path[1:]

        # descend through potential groups, creating some if they don't exist
        parts = path.split('/')

        # remove the last, deepest part of the path, which we will not be creating
        parts.pop()
        current_group = self._root_group
        while len(parts) > 0:
            expected_name = parts.pop(0)
            existing = [child for child in current_group.children if child.name == expected_name]
            if len(existing) == 0:
                # need to create this one
                current_group = current_group.add_group(expected_name)
            else:
                current_group = existing[0]

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

        parts = path.split('/')
        name = parts[-1]
        parent_elem = self._get_parent_from_path(path)._elem

        # make the new layer itself
        return self._add_layer(image, parent_elem, name, z_index=z_index, offsets=offsets, opacity=opacity, visible=visible,
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

        parts = path.split('/')
        name = parts[-1]
        parent_elem = self._get_parent_from_path(path)._elem

        # make the new group itself
        return self._add_group(parent_elem, name, z_index=z_index, offsets=offsets, opacity=opacity, visible=visible,
                        composite_op=composite_op, uuid=uuid, **kwargs)

    def remove(self, uuid):
        """
        Remove some layer or group and all of its children from the project
        :param path:
        :param uuid:
        :return:
        """
        
        root_child = self.get_by_uuid(uuid)

        children_to_remove = [root_child]
        if root_child.type == TYPE_GROUP:
            children_to_remove = children_to_remove + root_child.children_recursive

        parent_elem = root_child.parent._elem

        # remove all of the global references to uuids and elems
        for _child in children_to_remove:
            del self._children_elems[_child._elem]
            if _child.uuid is not None:
                del self._children_uuids[_child.uuid]

        # this should only have to be done for the parent for all of the other elements to be gone in the XML tree
        parent_elem.removeChild(root_child._elem)
        

    def move(self, src_uuid, dst_uuid, dst_z_index=1):
        """
        Move some layer or group and all of its children somewhere else inside the project
        If there are some layer groups that are missing for the destination to exist, they
        will be created automatically.
        
        :param uuid: source group/layer uuid to move        
        :param dest_uuid: dest group uuid to place source element inside of
        :param dest_z_index: inside of the destination group, place the moved layer/group at this index
        :return: None
        """

        if dst_uuid is None:
            dest_parent = self._root_group
        else:
            dest_parent = self.get_by_uuid(dst_uuid)

        child = self.get_by_uuid(src_uuid)

        # move elements first in the XML object repr, then the

        old_parent_elem = child.parent._elem
        old_parent_elem.removeChild(child._elem)
        self._insertElementAtIndex(dest_parent._elem, dst_z_index-1, child._elem)

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






