from PIL import Image
from pyora.Render import Renderer
from pyora import TYPE_GROUP, TYPE_LAYER


class OpenRasterItemBase:

    def __init__(self, project, elem, _type):
        self._elem = elem
        self._type = _type
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
        if self._elem is self._project._root_group._elem:
            return None
        parentNode = self._project._parentNode(self._elem)
        if parentNode is self._project._root_group._elem:
            return self._project._root_group
        return self._project._children_uuids[parentNode.attrib['uuid']]

    @property
    def is_group(self):
        return self.type == TYPE_GROUP

    @property
    def uuid(self):
        """
        :return: str - the layer uuid
        """
        return self._elem.attrib.get('uuid', None)

    @uuid.setter
    def uuid(self, value):
        self._project._children_uuids[str(value)] = self
        self._elem.set('uuid', str(value))

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
    def z_index(self):
        """
        Get the stacking position of the layer, relative to the group it is in (or the root group).
        Higher numbers are 'on top' of lower numbers. The lowest value is 1.
        :return: int - the z_index of the layer
        """
        if self.parent is None:
            return 1
        return list(reversed(self.parent._elem.getchildren())).index(self._elem) + 1

    @z_index.setter
    def z_index(self, new_z_index):
        """
        Reposition this layer inside of this group. (Uses 'relative' z_index)
        As with most z_index, 1 is the lowest value (painted first)
        :param new_z_index:
        :return:
        """
        if self.parent is None:
            raise ValueError("It is not possible to set the z-index of the root group")
        parent = self.parent._elem
        parent.remove(self._elem)
        parent.insert(len(parent) - (new_z_index - 1), self._elem)

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
    def visible_rendered(self):
        """
        :return: bool - visible property of this group when considering all ancestors
        """
        parent = self
        while True:
            if parent.hidden:
                return False
            parent = parent.parent
            if parent is None:
                return True

    @property
    def hidden_rendered(self):
        """
        :return: bool - hidden property of this group when considering all ancestors
        """
        return not self.visible_rendered

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


class Layer(OpenRasterItemBase):

    def __init__(self, image, project, elem):

        super().__init__(project, elem, TYPE_LAYER)

        self.image = image

    def __repr__(self):
        return f'<OpenRaster Layer "{self.name}" ({self.uuid})>'

    @OpenRasterItemBase.name.setter
    def name(self, value):
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

    def get_image_data(self, raw=True):
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

    @property
    def dimensions(self):
        """
        Not a supported ORA spec metadata, but we can read the specific PNG data to obtain the dimension value
        :return: (width, height) tuple of dimensions based on the content rect
        """
        return self.image.size


class Group(OpenRasterItemBase):
    def __init__(self, project, elem):

        super().__init__(project, elem, TYPE_GROUP)

    def __iter__(self):
        yield from self.children

    def __repr__(self):
        return f'<OpenRaster Group "{self.name}" ({self.uuid})>'

    def add_layer(self, image, name, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", uuid=None, **kwargs):
        return self._project._add_layer(image, self._elem, name, z_index=z_index, offsets=offsets,
                                        opacity=opacity, visible=visible, composite_op=composite_op,
                                        uuid=uuid, **kwargs)

    def add_group(self, name, z_index=1, offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", uuid=None, **kwargs):
        return self._project._add_group(self._elem, name, z_index=z_index, offsets=offsets,
                                        opacity=opacity, visible=visible, composite_op=composite_op,
                                        uuid=uuid, **kwargs)

    def add_tree(self, name, other_group):
        return self._project._add_tree(self._elem, name, other_group)

    @property
    def _renders_isolated(self):
        """
        should be isolated according to ORA spec
        """
        return self.isolated or \
               (self._project._isolate_non_opaque_groups and self.opacity < 1.0) or\
               self.composite_op != 'svg:src-over'

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
        self._elem.set('name', str(value))

    @property
    def children(self):
        """
        Returns all layers and groups under this group
        """
        for _child in self._elem:
            yield self._project.get_by_uuid(_child.attrib['uuid'])

    @property
    def children_recursive(self):
        """
        Returns all layers and groups under this group, and groups under that, etc.
        """
        for _child in self._elem.find('*'):
            yield self._project.get_by_uuid(_child.attrib['uuid'])

    @property
    def iter_tree(self):
        layer_list = []
        for layer in self._elem.iter():
            if layer.tag in ('layer', 'stack'):
                layer_list.insert(0, self._project._children_elems[layer])
        return layer_list

    @property
    def uuids(self):
        """
        Returns the uuids belonging to all layers and groups under this group
        """
        for _child in self._elem.find('*'):
            yield _child.attrib['uuid']

    @property
    def groups(self):
        """
        Returns the group objects under this group
        """
        for _child in self._elem.find('stack'):
            yield self._project.get_by_uuid(_child.attrib['uuid'])

    @property
    def layers(self):
        """
        Returns the layer objects under this group
        """
        for _child in self._elem.find('layer'):
            yield self._project.get_by_uuid(_child.attrib['uuid'])

    def get_image_data(self, raw=True):
        """
        Get a PIL Image() object of the group (composed of all underlying layers).
        """
        renderer = Renderer(self._project)
        _layerData = renderer.render(self)

        if raw:
            return _layerData
        dims = self._project.dimensions
        canvas = Image.new('RGBA', (dims[0], dims[1]))
        canvas.paste(_layerData, self.offsets)

        return canvas

    @property
    def dimensions(self):
        """
        Not a supported ORA spec metadata, but we can read all png data below us to find the total enclosed size
        """
        raise NotImplementedError()
        # iter layers below
        # for _item in self._elem.findall('.//*'):
        #     pass
        #     # for each item need to iterate downward until we hit a layer, keeping track of all of the x and y values
        #     # and totaling them up. Then for that layer we use its size to get the min/max x/y
        #
        #     # finally, out of all min/max x/y , get the greatest/least and return the differences
        #
        #     # min-x: the lowest value for offset
        # return self.image.size

