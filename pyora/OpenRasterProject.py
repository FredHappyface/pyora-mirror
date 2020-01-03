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
from pyora import TYPE_GROUP, TYPE_LAYER, ORA_VERSION


class OpenRasterItemBase:

    def __init__(self, project, parent, elem, path, _type):
        self._elem = elem
        self._parent = parent
        self._type = _type
        self._path = path
        self._project = project

    def __setitem__(self, key, value):
        """
        Set an arbitrary attribute on the underlying xml element
        """
        self._elem.set(key, value)

    def __contains__(self, item):
        return item in self._elem.attrib

    def __getitem__(self, item):
        return self._elem.attrib[item]

    @property
    def type(self):
        """
        :return
        """
        return self._type

    @property
    def parent(self):
        """
        Get the group object for the parent of this layer or group
        :return:
        """
        return self._parent

    @property
    def is_group(self):
        return self.type == TYPE_GROUP

    @property
    def UUID(self):
        """
        :return: str - the layer UUID
        """
        return self._elem.attrib.get('UUID', None)

    @UUID.setter
    def UUID(self, value):
        self._project._children_uuids[str(value)] = self
        self._elem.set('UUID', str(value))

    @property
    def name(self):
        """
        :return: str - the layer name
        """
        return self._elem.attrib.get('name', None)

    @name.setter
    def name(self, value):

        self._elem.set('name', str(value))

    @property
    def path(self):
        """
        :return: str - the layer path
        """
        return self._path

    @property
    def z_index(self):
        """
        Get the stacking position of the layer, relative to the group it is in (or the root group).
        Higher numbers are 'on top' of lower numbers. The lowest value is 1.
        :return: int - the z_index of the layer
        """
        return list(reversed(self._parent._elem.getchildren())).index(self._elem) + 1

    @z_index.setter
    def z_index(self, new_z_index):
        """
        Reposition this layer inside of this group. (Uses 'relative' z_index)
        As with most z_index, 1 is the lowest value (painted first)
        :param new_z_index:
        :return:
        """


    @property
    def visible(self):
        """
        :return: bool - is the layer visible
        """
        return self._elem.attrib.get('visibility', 'visible') == 'visible'

    @visible.setter
    def visible(self, value):
        self._elem.set('visibility', 'visible' if value else 'hidden')

    @property
    def hidden(self):
        """
        :return: bool - is the layer hidden
        """
        return not self.visible

    @hidden.setter
    def hidden(self, value):
        self._elem.set('visibility', 'hidden' if value else 'visible')

    @property
    def opacity(self):
        """
        :return: float 0.0 - 1.0 defining opacity
        """
        try:
            return float(self._elem.attrib.get('opacity', '1'))
        except:
            print(f"Malformed value for opacity {self}, defaulting to 1.0")
            return 1.0

    @opacity.setter
    def opacity(self, value):
        self._elem.set('opacity', str(float(value)))

    @property
    def offsets(self):
        """
        :return: (left, top) starting coordinates of the top left corner of the png data on the canvas
        """
        try:
            return int(self._elem.attrib.get('x', '0')), int(self._elem.attrib.get('y', '0'))
        except:
            print(f"Malformed value for offsets {self}, defaulting to 0, 0")
            return 0, 0

    @offsets.setter
    def offsets(self, value):
        self._elem.set('x', str(value[0]))
        self._elem.set('y', str(value[1]))

    @property
    def dimensions(self):
        """
        Not a supported ORA spec metadata, but we can read the specific PNG data to obtain the dimension value
        :return: (width, height) tuple of dimensions based on the content rect
        """
        raise NotImplementedError()

    @property
    def bounding_rect(self):
        """
        Not a supported ORA spec metadata, but we can read the specific PNG data to obtain the dimension value
        :return: (left, top, right, bottom) tuple of content rect
        """
        raise NotImplementedError()

    @property
    def raw_attributes(self):
        """
        Get a dict of key:value pairs of xml attributes for the element defining this object
        Useful if something is not yet defined as a method in this library
        :return: dict of attributes
        """
        return self._elem.attrib

    @property
    def composite_op(self):
        """
        :return: string of composite operation intended for the layer / group
        """
        return self._elem.attrib.get('composite-op', None)

    @composite_op.setter
    def composite_op(self, value):
        self._elem.set('composite-op', str(value))

class Group(OpenRasterItemBase):
    def __init__(self, project, parent, elem, path):

        super().__init__(project, parent, elem, path, TYPE_GROUP)

        self._layers = []
        self._layers_names = {}
        self._layers_uuids = {}

        self._groups = []
        self._groups_names = {}
        self._groups_uuids = {}

        self._children = []
        self._children_names = {}
        self._children_uuids = {}

    def __iter__(self):
        yield from reversed(self._children)

    def __repr__(self):
        return f'<OpenRaster Group "{self.name}" ({self.UUID})>'

    @property
    def isolated(self):
        """
        :return: bool - is the layer rendered isolated
        """
        return self._elem.attrib.get('isolation', 'auto') == 'isolate'

    @isolated.setter
    def isolated(self, value):
        """
        Set the isolation rendering property of this group.
        By default, groups are isolated, which means that composite and blending will be performed as if
        the group was over a blank background. Other layers painted below the group are not composited/blended with
        (until the whole group is done rendering by itself, at which point it is composited/blended with its own
        composite-op attribute to the painted canvas below it) If isolation is turned off, the base background will
        be the current canvas already painted, instead of a blank canvas.
        To comply with ORA spec, the isolation property is ignored (and groups are forced to be rendered isolated)
        if either (1) their opacity is less than 1.0 or (2) they use a composite-op other than 'svg:src-over'
        :param value:
        :return:
        """
        self._elem.set('isolation', 'isolate' if value else 'auto')

    @OpenRasterItemBase.name.setter
    def name(self, value):

        old_path = self._path
        parts = self._path.split('/')
        parts[-1] = value
        self._path = '/'.join(parts)

        # in this case we also need to go through all the other paths that involved this group and replace them
        for _path in self._project._children_paths:
            if _path.startswith(old_path):
                _new_path = _path.replace(old_path, self._path, 1)
                self._project._children_paths[_new_path] = self._project._children_paths.pop(_path)

        self._elem.set('name', str(value))

    def _add_child(self, child):
        if child.type == TYPE_GROUP:
            self._groups.append(child)
            self._groups_names[child.name] = child
            self._groups_uuids[child.UUID] = child
        if child.type == TYPE_LAYER:
            self._layers.append(child)
            self._layers_names[child.name] = child
            self._layers_uuids[child.UUID] = child
        self._children.append(child)
        self._children_names[child.name] = child
        self._children_uuids[child.UUID] = child

    @property
    def layers(self):
        return self._layers

    @property
    def groups(self):
        return self._groups

    def get_image_data(self, raw=False):
        """
        Get a PIL Image() object of the group (composed of all underlying layers).
        By default the returned image will always be the same dimension as the project canvas, and the original
        image data will be placed / cropped inside of that.
        :param raw: Instead of cropping to canvas, just get the image data exactly as it exists
        :return: PIL Image()
        """
        raise NotImplementedError()


class Layer(OpenRasterItemBase):

    def __init__(self, image, project, parent, elem, path):

        super().__init__(project, parent, elem, path, TYPE_LAYER)

        self.image = image

    def __repr__(self):
        return f'<OpenRaster Layer "{self.name}" ({self.UUID})>'

    @OpenRasterItemBase.name.setter
    def name(self, value):

        # need to update stored paths in parent
        old_path = self._path
        parts = self._path.split('/')
        parts[-1] = value
        self._path = '/'.join(parts)
        self._project._children_paths[self._path] = self._project._children_paths.pop(old_path)

        self._elem.set('name', str(value))

    def _set_image_data(self, image):
        self.image = image

    def set_image_data(self, image):
        """
        Change the image data for this layer
        :param image: pil Image() object of the new layer
        :return: None
        """
        self._set_image_data(image)

    def get_image_data(self, raw=False):
        """
        Get a PIL Image() object of the layer.
        By default the returned image will always be the same dimension as the project canvas, and the original
        image data will be placed / cropped inside of that.
        :param raw: Instead of cropping to canvas, just get the image data exactly as it exists
        :return: PIL Image()
        """


        _layerData = self.image

        if raw:
            return _layerData
        dims = self._project.dimensions
        canvas = Image.new('RGBA', (dims[0], dims[1]))
        canvas.paste(_layerData, self.offsets)

        return canvas

    @property
    def z_index_global(self):
        """
        Get the stacking position of the layer, relative to the entire canvas.
        Higher numbers are 'on top' of lower numbers. The lowest value is 1.
        :return: int - the z_index of the layer
        """
        for i, layer in enumerate(self._project, 1):
            if layer == self:
                return i
        assert False  # should never not find a latching layer...




class Project:

    def __init__(self):
        self._children = []
        self._children_paths = {}
        self._children_elems = {}
        self._children_uuids = {}
        self._extracted_merged_image = None


    def __iter__(self):
        for layer in reversed(self._elem_root.findall('.//layer')):
            yield self._children_elems[layer]

    @property
    def iter_layers(self):
        return self.__iter__()

    @property
    def iter_groups(self):
        for group in reversed(self._elem_root.findall('.//stack')):
            if group == self._root_group._elem:
                yield self._root_group
            else:
                yield self._children_elems[group]

    def _zip_store_image(self, zipref, path, image):
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        imgByteArr.seek(0)
        zipref.writestr(path, imgByteArr.read())

    @staticmethod
    def extract_layer(path_or_file, path=None, UUID=None, pil=False):
        """
        Efficiently extract just one specific layer image
        :param path_or_file: Path to ORA file or file handle
        :param path: Path of layer to extract in the ORA file
        :param UUID: UUID of layer to search for in the ORA file (if path not provided)
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
                    _elem_root = _elem_root.find(f".//layer[@uuid='{UUID}']")
                    if not _elem_root:
                        raise ValueError("Unable to find layer with UUID %s in ORA!" % UUID)

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

                    cur_path = basepath + '/' + child_elem.attrib['name']
                    if child_elem.tag == 'stack':
                        _new = Group(self, parent, child_elem, cur_path)
                        parent._add_child(_new)
                        _build_tree(_new, cur_path)
                    elif child_elem.tag == 'layer':
                        with zipref.open(child_elem.attrib['src']) as layerFile:
                            image = Image.open(layerFile)
                        _new = Layer(image, self, parent, child_elem, cur_path)
                        parent._add_child(_new)
                    else:
                        print(f"Warning: unknown tag in stack: {child_elem.tag}")
                        continue

                    self._children.append(_new)

                    self._children_paths[cur_path] = _new
                    self._children_elems[child_elem] = _new
                    self._children_uuids[_new.UUID] = _new

            self._root_group = Group(self, None, self._elem, '/')
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
        self._root_group = Group(self, None, self._elem, '/')
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

    def _add_elem(self, tag, path, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True, composite_op="svg:src-over",
                  **kwargs):

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
        index = len([x for x in self.children if x.type == TYPE_LAYER])
        new_filename = f'/data/layer{index}.png'

        # add xml element
        elem = self._add_elem('layer', path, **kwargs, src=new_filename)
        obj = Layer(image, self, self._get_parent_from_path(path), elem, path)
        obj._parent._add_child(obj)

        self.children.append(obj)
        self._children_paths[path] = obj
        self._children_elems[elem] = obj
        self._children_uuids[obj.UUID] = obj

        return obj

    # def delete_path(self, path):
    #     item = self._children_paths[path]
    #     del self._children_paths[path]
    #     del self._children_elems[item._elem]
    #     if item.UUID:
    #         del self._children_uuids[item.UUID]
    #     self.children.remove(item)

    def _add_group(self, path, **kwargs):
        elem = self._add_elem('stack', path, **kwargs)
        obj = Group(self, self._get_parent_from_path(path), elem, path)
        obj._parent._add_child(obj)

        if not 'isolation' in kwargs:
            kwargs['isolation'] = 'isolate'

        self.children.append(obj)
        self._children_paths[path] = obj
        self._children_elems[elem] = obj
        self._children_uuids[obj.UUID] = obj
        return obj

    def _make_groups_recursively(self, path):

        # absolute path slash is for styling/consistency only, remove it if exists
        if path[0] == '/':
            path = path[1:]

        # determine if the required group exists yet
        # and add all required groups to make the needed path
        parts = path.split('/')
        parent_path = '/'.join(parts[:-1])
        if not parent_path in self._children_paths:
            for i, _parent_name in enumerate(parts[1:], 1):
                _sub_parent_path = '/' + '/'.join(parts[:i])
                if not _sub_parent_path in self._children_paths:
                    # make new empty group
                    self._add_group(_sub_parent_path, isolation='isolate')

    def add_layer(self, image, path=None, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", UUID=None, **kwargs):
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
                        composite_op=composite_op, UUID=UUID, **kwargs)

    def add_group(self, path, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", UUID=None, isolated=True, **kwargs):
        """
        Append a new layer group to the project
        :param path: Absolute filesystem-like path of the group in the project. For example "/group1" or
        "/group1/group2". If given without a leading slash, like "group3", we assume the group is placed at
        the root of the project.
        :param offsets: tuple of (x, y) offset from the top-left corner of the Canvas
        :param opacity: float - group opacity 0.0 to 1.0
        :param visible: bool - is the group visible
        :param composite_op: str - composite operation attribute passed directly to stack / layer element
        :param UUID: str - UUID identifier value for this group
        :param isolation:bool - True or False
        :return: Layer() - reference to the newly created layer object
        """
        self._make_groups_recursively(path)

        if not path[0] == '/':
            path = '/' + path

        kwargs['isolation'] = 'isolate' if isolated else 'auto'

        # make the new group itself
        return self._add_group(path, z_index=z_index, offsets=offsets, opacity=opacity, visible=visible,
                        composite_op=composite_op, UUID=UUID, **kwargs)


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
    def UUIDs(self):
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

        return make_merged_image(self)

    def get_thumbnail_image_data(self, use_original=False):
        """
        Get a PIL Image() object of the entire project (composite) (standard 256x256 max ORA thumbnail size
        :param use_original: IF true, and If there was a stored 'mergedimage' already in the file which was opened,
        just return that. In any other case a new merged image is generated.
        :return: PIL Image()
        """
        if self._extracted_merged_image and use_original:
            return make_thumbnail(self._extracted_merged_image)

        return make_thumbnail(make_merged_image(self))






