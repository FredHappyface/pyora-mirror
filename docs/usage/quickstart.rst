Quickstart
=======================================

Quick tutorial for easy usage of pyora.

General Info
=======================================
    - Opacity values should always be 0.0 - 1.0
    - By default / standard, each layer is the same size as the whole project. This seems to be by design.

Limitations
=======================================
    - Creating the "mergedimage" component of the ORA file (for preview) currently only supports "svg:src-over" (simple) compositing. So it is recommended to provide your own composite image file for the most accurate previews.
    - The mergedimage limitations also apply to thumbnails
    - When modifying an existing file in place, currently only adding / modifying layers is supported. Moving in the tree and deleting soon to come.


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
When adding layers, the z_index always refers the offset in the group you are putting the layer in.

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



.. _Pillow/PIL Image() Object: https://pillow.readthedocs.io/en/stable/reference/Image.html