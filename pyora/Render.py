from PIL import Image


def make_merged_image(project):
    canvas = Image.new('RGBA', project.dimensions)
    for layer in project:
        if layer.visible:
            base_img = layer.get_image_data()
            base_img.putalpha(int(255*layer.opacity))
            canvas.paste(layer.get_image_data(), layer.offsets)
    return canvas


def make_thumbnail(image):
    # warning: in place modification
    if image.size[0] > 256 or image.size[1] > 256:
        image.thumbnail((256, 256))
