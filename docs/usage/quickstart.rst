Quickstart
=======================================

Quick tutorial for easy usage of pyora.

General Info
=======================================
    - Opacity values should always be 0.0 - 1.0
    - By default / standard, each layer is the same size as the whole project. This seems to be by design.

Limitations
=======================================
    - The implementations for the non-separable blend modes (hue, color, saturation, luminosity) are currently non-vectorized and quite slow
    - Making group opacity <1.0 for non-isolated groups is currently undefined by the standard, but seems to be defined in the main programs using the standard. By default, we can not guarantee the rendering in this case. If you would like to strictly follow the standard in this regard (groups with <1.0 opacity become isolated), you may set:
        - <instance of Project()>._isolate_non_opaque_groups = True


Reading Data
=======================================

Available data for reading includes basic metadata about the project and it's groups/layers, as well as the Image data.

::

    from pyora import Project, TYPE_LAYER
    from PIL import Image

    project = Project.load("/home/pjewell/Documents/yay3.ora")
    width, height = project.dimensions
    print(width, height)

    # layers can be referenced in order
    for layer in project.children:
        if layer.type == TYPE_LAYER:
            print(layer.name, layer.UUID)
            print(layer.z_index, layer.z_index_global, layer.opacity, layer.visible, layer.hidden)

    # or by path
    layer = project['/Layer 2']
    print(layer)

    # or by UUID
    layer = project.get_by_uuid("6A163DC4-B344-4C76-AF3B-369484692B20")
    print(layer)

You can get image data for each layer separately. Each image data will be returned as a
`Pillow/PIL Image() Object`_.

::

    # layers can be referenced in order
    for layer in project.children:
        if layer.type == TYPE_LAYER:
            layer.get_image_data().crop((0,0,50,50)).save(f"{layer.name}.png")  # all PIL methods work here

Making a new project / Writing Data
=======================================

This is very simple, just use Project.new() / Project.save()

::

    # make a new ORA project, numbers are canvas dimensions
    project = Project.new(100, 100)

    # adding a layer
    project.add_layer(Image.open("<path to some image file>"), 'Root Level Layer')
    # adding a layer at any arbitrary path will automatically create the appropriate layer groups
    project.add_layer(Image.open("<path to some image file>"), 'Some Group/Another Layer')

    # two ways to set attributes, during add and after the fact
    new_layer = project.add_layer(Image.open("<path to some image file>"), 'Group2/Layer3',
                               opacity=0.5, offsets=(10, 20,))
    new_layer.opacity = 0.86

    # set arbitrary future attributes before they are officially supported
    new_layer["awesome-future-thing"] = 'yes'

    # you can add groups manually too if you like, though its not usually required
    project.add_group(path='/manually added group')

    # save current project as ORA file
    project.save("<path to some .ora save location>")


Z_indexes
=======================================

You can get the z_index of layers relative to the group they are in, or globally in the whole project.
When adding layers, you have three choices for z_index:
    -'above' , which places the layer "above" (painted on top of) all other layers in that group
    -'below' , which places the layer "below" all other layers in that group
    -some integer, **1 indexed**, to specify where to insert the layer. (for example, z_index=1 is the same as z_index='below')

::

    project = Project.new(100, 100)
    img = Image.open("/home/pjewell/Pictures/Screenshot_20191105_113922.png")
    project.add_layer(img, 'g1/l1')
    g1l2 = project.add_layer(img, 'g1/l2')
    project.add_layer(img, 'g2/l1')
    g2l2 = project.add_layer(img, 'g2/l2')
    project.add_layer(img, 'g1/l3', z_index=2)  # z_index is relative to group, l3 should be in between l1 and l2
    l0 = project.add_layer(img, 'l0', z_index=2)  # l0 should be at the root of the project in between groups g1 and g2

    print(g1l2.z_index, g1l2.z_index_global)  # index 3 in the group, third layer overall
    print(g2l2.z_index, g2l2.z_index_global)  # index 2 in the group, sixth layer overall
    print(l0.z_index, l0.z_index_global)  # index 2 in the group (the root group), fourth layer overall

    project.save("<path to some .ora save location>")

Manual Rendering
=======================================

You can render together some layers using the composite operators, without the overhead of exporting a whole project.

::


    from pyora import Renderer

    project = Project.new(200, 200)
    project.add_layer(Image.open("/home/pjewell/Pictures/Screenshot_20191105_113922.png"), 'g1/l1')
    project.add_layer(Image.open("/home/pjewell/Pictures/192-1.jpg"), 'g1/l2', composite_op='svg:color-burn')
    r = Renderer(project)
    final = r.render()  # returns PIL Image()
    final.save(f'/home/pjewell/Pictures/composite_quick.png')

    # optional, for comparison
    project.save(f'/home/pjewell/Pictures/composite_quick.ora')

.. _Pillow/PIL Image() Object: https://pillow.readthedocs.io/en/stable/reference/Image.html