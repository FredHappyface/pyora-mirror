import io
import zipfile
from PIL import Image
import defusedxml.ElementTree as ET
from xml.etree.ElementTree import Element
from pyora.Render import Renderer, make_thumbnail
from pyora.Layer import Layer, Group
from pyora import TYPE_GROUP, TYPE_LAYER, ORA_VERSION
import re
import uuid
from copy import deepcopy
from contextlib import ExitStack

class Project:

    def __init__(self):
        self._children = []
        self._children_elems = {}
        self._children_uuids = {}
        self._extracted_merged_image = None
        self._generated_uuids = False
        self._isolate_non_opaque_groups = False

    def get_by_path(self, path):
        """Find a group or layer object using a *Nix-like 'path', based on the names of project groups / layers

        Paths look like '/group1/layer2', note that while this provides good ability to extract any specific layer
        in most circumstances, it does not work well in the case that there are multiple groups / layers in the same
        group, with the same name. In this case, you will need to drill down by using .children(), or alternatively
        getting by uuid with get_by_uuid()

        Returns:
            pyora.Group or pyora.Layer: the found layer object
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
        """Get a reference to the outermost layer group containing everything else

        Returns:
            pyora.Group: root Group() Object
        """
        return self._root_group

    @property
    def uuids(self):
        return self._children_uuids

    @property
    def iter_tree(self):
        layer_list = []
        for layer in self._elem_root.iter():
            if layer.tag in ('layer', 'stack'):
                layer_list.insert(0, self._children_elems[layer])
        return layer_list

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
        """Efficiently extract just one specific layer image

        This method extracts just the image data of the thumbnail, without reading the entire ORA file.
        Specify either a path or uuid, not both.

        Args:
            path_or_file (str, File-Like object) : filesystem path or .read()-able file object of ORA file to load
            path (str): Path of layer to extract in the ORA file
            uuid (str): uuid of layer to search for in the ORA file (if path not provided)
            pil (bool): for consistency, if true, wrap the image with PIL and return PIL.Image() object

        Returns:
            Bytes or PIL.Image(): Depends on 'pil' argument
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
        """Efficiently extract just existing composite full resolution image in the ORA file

        This method extracts just the image data of the thumbnail, without reading the entire ORA file

        Args:
            path_or_file (str, File-Like object) : filesystem path or .read()-able file object of ORA file to load
            pil (bool): for consistency, if true, wrap the image with PIL and return PIL.Image() object

        Returns:
            Bytes or PIL.Image(): Depends on 'pil' argument
        """
        with zipfile.ZipFile(path_or_file, 'r') as zipref:
            with zipref.open('mergedimage.png') as imgdata:
                if pil:
                    return Image.open(imgdata)
                return imgdata.read()

    @staticmethod
    def extract_thumbnail(path_or_file, pil=False):
        """Efficiently extract just the thumbnail image

        This method extracts just the image data of the thumbnail, without reading the entire ORA file

        Args:
            path_or_file (str, File-Like object) : filesystem path or .read()-able file object of ORA file to load
            pil (bool): for consistency, if true, wrap the image with PIL and return PIL.Image() object

        Returns:
            Bytes or PIL.Image(): Depends on 'pil' argument
        """
        with zipfile.ZipFile(path_or_file, 'r') as zipref:
            with zipref.open('Thumbnails/thumbnail.png') as imgdata:
                if pil:
                    return Image.open(imgdata)
                return imgdata.read()

    @staticmethod
    def load(path_or_file):
        """Load an existing ORA file into a pyora project

        Factory function. will instantiate and return a new pyora.Project() when called.

        Args:
            path_or_file (str, File-Like object) : filesystem path or .read()-able file object of ORA file to load

        Returns:
            pyora.Project : the new instance
        """
        proj = Project()
        proj._load(path_or_file)
        return proj

    def from_stack_xml(self, xml_dom, srcs_to_files):
        """Overwrite the current pyora project instance directly from XML stack + image data sources

        This is the minimum amount of data needed to create an ORA project, if perhaps storing ORA files is not
        as convenient for your use case.

        Args:
            xml_dom (str, ET): either a string of xml or the parsed XML elementtree instance
            srcs_to_files (dict): dict of layer "src" attribute strings, to .read()-able File-like objects of
                of the layer PNG (or other image) data. Any images that PIL supports are fine.

        """
        if type(xml_dom) is str:
            xml_dom = ET.fromstring(xml_dom)

        self._load(path_or_file=None, xml_dom=xml_dom, srcs_to_files=srcs_to_files)

    def _load(self, path_or_file=None, xml_dom=None, srcs_to_files=None):
        """
        Can either be called with "path_or_file" to open an existing ORA file, or with
        both "xml_dom" and "uuid_to_files" specified
        """

        # these three lines basically make a conditional 'with' statement
        # (we only need the zip file stack context if xml_dom is None)
        # https://stackoverflow.com/a/34798330/2205380
        with ExitStack() as stack:
            if not xml_dom:
                zipref = stack.enter_context(zipfile.ZipFile(path_or_file, 'r'))


            self._children = []
            self._children_elems = {}
            self._children_uuids = {}

            if xml_dom:
                self._extracted_merged_image = None
                self._elem_root = xml_dom
            else:

                with zipref.open('mergedimage.png') as mergedimage:
                    self._extracted_merged_image = Image.open(mergedimage).copy()

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

            def _build_tree(parent):

                for child_elem in parent._elem:
                    if not child_elem.attrib.get('uuid', None):
                        self._generated_uuids = True
                        child_elem.set('uuid', str(uuid.uuid4()))

                    if child_elem.tag == 'stack':
                        _new = Group(self, child_elem)
                        _build_tree(_new)
                    elif child_elem.tag == 'layer':
                        if xml_dom:
                            image = Image.open(srcs_to_files[child_elem.attrib.get('src')]).convert('RGBA')
                        else:
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


            self._children_elems[self._elem] = self._root_group


            _build_tree(self._root_group)


    @staticmethod
    def new(width, height, xres=72, yres=72):
        """Start a blank new ORA project

        Factory function. will instantiate and return a new pyora.Project() when called.

        Args:
            width (int): initial width of canvas, in px
            height (int): initial height of canvas, in px
            xres (int): nominal resolution pixels per inch in x
            yres (int): nominal resolution pixels per inch in y

        Returns:
            pyora.Project : the new instance
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

        self._children_elems[self._elem] = self._root_group


        self._extracted_merged_image = None

    def save(self, path_or_file, composite_image=None, use_original=False):
        """Save the current project state to an ORA file.

        Args:
            path (str): path to the ora file to save
            composite_image (PIL.Image()): PIL Image() object of the composite rendered canvas. It is used to
                create the mergedimage full rendered preview, as well as the thumbnail image. If not provided,
                one will be generated by pyora's Render() class by stacking all of the layers in the project.
                Note that the image you pass may be modified during this process, so if you need to use it elsewhere
                in your code, you should copy() first.
            use_original (bool): If true, and If there was a stored 'mergedimage' already in the file which was opened,
                use that for the 'mergedimage' in the new file, instead of rendering a new one.
        """
        with zipfile.ZipFile(path_or_file, 'w') as zipref:

            zipref.writestr('mimetype', "image/openraster".encode())

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

            filename_counter = 0
            for layer in self.children_recursive:
                if layer.type == TYPE_LAYER:
                    new_filename = f'/data/layer{filename_counter}.png'
                    layer._elem.attrib['src'] = new_filename
                    filename_counter += 1
                    self._zip_store_image(zipref, layer['src'], layer.get_image_data(raw=True))

            zipref.writestr('stack.xml', ET.tostring(self._elem_root, method='xml'))

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

    def _resolve_z_index(self, parent_elem, z_index):
        if z_index == 'above':
            return 0
        elif z_index == 'below':
            return len(parent_elem)
        elif z_index >= 1:
            # xml library does not mind negative numbers, but we check anyway to be safe
            return max(0, len(parent_elem) - (z_index - 1))
        else:
            return 0

    def _add_elem(self, tag, parent_elem, name, z_index='above', offsets=(0, 0,), opacity=1.0, visible=True, composite_op="svg:src-over",
                  **kwargs):

        if tag == 'stack' and not 'isolated' in kwargs:
            kwargs['isolated'] = True

        if not 'uuid' in kwargs or kwargs['uuid'] is None:
            self._generated_uuids = True
            kwargs['uuid'] = str(uuid.uuid4())

        new_elem = Element(tag, {'name': name, 'x': str(offsets[0]), 'y': str(offsets[1]),
                                        'visibility': 'visible' if visible else 'hidden',
                                        'opacity': str(opacity), 'composite-op': composite_op,
                                    **{k: str(v) for k, v in kwargs.items() if v is not None}})

        parent_elem.insert(self._resolve_z_index(parent_elem, z_index), new_elem)

        return new_elem

    def _add_layer(self, image, parent_elem, name, **kwargs):

        # add xml element
        elem = self._add_elem('layer', parent_elem, name, **kwargs)
        obj = Layer(image, self, elem)

        self._children.append(obj)
        self._children_elems[elem] = obj
        self._children_uuids[obj.uuid] = obj

        return obj

    def _add_group(self, parent_elem, name, **kwargs):
        if not 'isolation' in kwargs:
            kwargs['isolation'] = 'isolate'

        elem = self._add_elem('stack', parent_elem, name, **kwargs)
        obj = Group(self, elem)

        self._children.append(obj)
        self._children_elems[elem] = obj
        self._children_uuids[obj.uuid] = obj
        return obj

    def _add_tree(self, parent_elem, name, other_group):
        """
        Add a group, recursively, under the specified parent
        Each element is copied and has it's attributes copied.
        """

        def _build_tree(parent):
            for child_elem in reversed(parent._elem):
                if not child_elem.attrib.get('uuid', None) or child_elem.attrib['uuid'] in self._children_uuids:
                    child_elem.attrib['uuid'] = str(uuid.uuid4())

                if child_elem.tag == 'stack':
                    _new = Group(self, child_elem)
                    _build_tree(_new)
                elif child_elem.tag == 'layer':
                    image = other_group._project.get_by_uuid(child_elem.attrib['uuid']).get_image_data(raw=True)
                    _new = Layer(image, self, child_elem)
                else:
                    print(f"pyora warning: Unknown tag in stack: {child_elem.tag}")
                    continue
                self._children.append(_new)
                self._children_elems[child_elem] = _new
                self._children_uuids[_new.uuid] = _new

        # insert XML structure
        cloned_xml = deepcopy(other_group._elem)
        z_index = 1

        cloned_xml.attrib['name'] = name
        if not cloned_xml.attrib.get('uuid', None) or cloned_xml.attrib['uuid'] in self._children_uuids:
            cloned_xml.attrib['uuid'] = str(uuid.uuid4())

        self._insertElementAtIndex(parent_elem, len(parent_elem) - (z_index-1), cloned_xml)

        new_outer_group = Group(self, cloned_xml)

        self._children.append(new_outer_group)
        self._children_elems[cloned_xml] = new_outer_group
        self._children_uuids[new_outer_group.uuid] = new_outer_group

        _build_tree(new_outer_group)

        return new_outer_group

    def _make_groups_recursively(self, path):
        """
        creates all of the groups which would be required UNDER the specified path (not the final, deepest path element)
        as this works with paths it will just choose the first matching path if duplicate names are found
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


    def add_layer(self, image, path=None, z_index='above', offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", uuid=None, **kwargs):
        """Append a new layer to the project

        Args:
            image (PIL.Image()): a PIL Image() object containing the image data to add
            path (str): Absolute filesystem-like path of the group in the project. For example "/group1" or
                "/group1/group2". If given without a leading slash, like "group3", we assume the group is placed at
                the root of the project.
            z_index (str, int): the index to place the new layer in inside of the group. 'above' places the layer at the
                top of the group. 'below' places the layer at the very bottom of the group. Other numbers (1 indexed)
                place the layer at that z_index, similar to css z-indices.
            offsets (tuple[int]): tuple of (x, y) offset (in px) from the top-left corner of the Canvas
            opacity (float): group opacity 0.0 to 1.0
            visible (bool): is the group visible (hidden if false)
            composite_op (str): composite operation attribute passed directly to stack / layer element (see blend
                modes documentation)
            uuid (str): uuid identifier value for this group

        Returns:
            pyora.Layer: reference to the newly created layer object
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

    def add_group(self, path, z_index='above', offsets=(0, 0,), opacity=1.0, visible=True,
                  composite_op="svg:src-over", uuid=None, isolated=True, **kwargs):
        """Append a new layer group to the project

        The group added this way starts out with no children (empty)

        Args:
            path (str): Absolute filesystem-like path of the group in the project. For example "/group1" or
                "/group1/group2". If given without a leading slash, like "group3", we assume the group is placed at
                the root of the project.
            z_index (str, int): the index to place the new layer in inside of the group. 'above' places the layer at the
                top of the group. 'below' places the layer at the very bottom of the group. Other numbers (1 indexed)
                place the layer at that z_index, similar to css z-indices.
            offsets (tuple[int]): tuple of (x, y) offset (in px) from the top-left corner of the Canvas
            opacity (float): group opacity 0.0 to 1.0
            visible (bool): is the group visible (hidden if false)
            composite_op (str): composite operation attribute passed directly to stack / layer element (see blend
                modes documentation)
            uuid (str): uuid identifier value for this group
            isolation (bool): Is the group isolated (composited separately from the groups below it)

        Returns:
            pyora.Layer: reference to the newly created group layer object
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
        """Remove a layer or group

        Args:
            uuid (str): The UUID of the layer or group to remove
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
        parent_elem.remove(root_child._elem)
        

    def move(self, src_uuid, dst_uuid, dst_z_index='above'):
        """Reposition a layer or group

        Move some layer or group and all of its children somewhere else inside the project
        If there are some layer groups that are missing for the destination to exist, they
        will be created automatically.

        Args:
            uuid (str): source group/layer uuid to move
            dest_uuid (str): dest group uuid to place source element inside of
            dest_z_index (str, int): the index to place the new layer in inside of the group. 'above' places the layer at the
                top of the group. 'below' places the layer at the very bottom of the group. Other numbers (1 indexed)
                place the layer at that z_index, similar to css z-indices

        """

        if dst_uuid is None:
            dest_parent = self._root_group
        else:
            dest_parent = self.get_by_uuid(dst_uuid)

        child = self.get_by_uuid(src_uuid)

        # move elements first in the XML object repr, then the

        old_parent_elem = child.parent._elem
        old_parent_elem.remove(child._elem)
        self._insertElementAtIndex(dest_parent._elem, self._resolve_z_index(dest_parent._elem, dst_z_index), child._elem)

    @property
    def dimensions(self):
        """Get Project (width, height) dimensions in px

        Returns:
            tuple[int]: (width, height) of project, in px
        """
        return int(self._elem_root.attrib['w']), int(self._elem_root.attrib['h'])

    @dimensions.setter
    def dimensions(self, size):
        """Set the dimensions of the project canvas

        Args:
             size (tuple): Tuple of new (width, height) for project canvas
        """
        self._elem_root.attrib['w'] = str(int(size[0]))
        self._elem_root.attrib['h'] = str(int(size[1]))

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
        """Get a PIL Image() object of the entire project (composite)

        Args:
            use_original (bool): If true, and If there was a stored 'mergedimage' already in the file which was opened,
                just return that. In any other case a new merged image is generated.

        Returns:
            PIL Image(): Image object
        """

        if self._extracted_merged_image and use_original:
            return self._extracted_merged_image

        r = Renderer(self)
        return r.render()

    def get_thumbnail_image_data(self, use_original=False):
        """Get the thumbnail image for the ora file

        Get PIL Image() object, composite, resized to standard 256x256 max ORA thumbnail size.

        Args:
            use_original (bool): If true, and If there was a stored 'mergedimage' already in the file which was opened,
                just return that. In any other case a new merged image is generated.

        Returns:
            PIL Image(): Image object
        """
        if self._extracted_merged_image and use_original:
            return make_thumbnail(self._extracted_merged_image)

        r = Renderer(self)
        return make_thumbnail(r.render())

    def get_stack_xml(self):
        """Get the current stack xml representation of the project

        Can be saved for later use with Project.set_stack_xml(), equivalent to the 'stack.xml' which would be
        saved on Project.save()

        Returns:
            str: string of stack xml
        """
        return ET.tostring(self._elem_root, method='xml')

    def set_stack_xml(self, xml_dom, new_sources=None):
        """Set the current stack xml representation of the project

        Using the 'stack.xml' standard format, update the current project to reflect the new scheme.
        This allows updating attributes / positioning in an exportable format without needing to
        store / transfer all of the data in the raster files themselves. Important! all uuids in
        the incoming scheme must match uuids in the current project, or undefined behavior fill occur.

        Args:
            xml_dom (str, ET()): String or parsed elementtree() of the stack.xml standard format
            new_sources (dict): (optional) dict of {uuid: new Image() object} of raster image sources
                to update during application of the new xml stack
        """

        if type(xml_dom) is str:
            xml_dom = ET.fromstring(xml_dom)

        # replace the xml doc with the new incoming one
        self._elem_root = xml_dom

        # update the fixed root references
        self._elem = self._elem_root[0]
        self._root_group._elem = self._elem
        self._children_elems = {}

        # find every relevant element in the new document, and update references to new elems
        for _child in self._elem.iter():
            obj = self._children_uuids[_child.attrib['uuid']]
            self._children_uuids[_child.attrib['uuid']]._elem = _child
            self._children_elems[_child] = obj

        # if updated sources are provided, apply them
        if new_sources:
            for _uuid in new_sources:
                if not _uuid in self._children_uuids:
                    print(f"Pyora Warning: Not able to set new source for UUID {_uuid} ;"
                          f" it does not exist in the new tree")
                    continue
                self._children_uuids[_uuid].src = new_sources[_uuid]