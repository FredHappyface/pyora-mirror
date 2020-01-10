from PIL import Image

"""
Blend modes from the spec:

Value	        Blending function	     Compositing Operator
svg:src-over	Normal	                 Source Over
svg:multiply	Multiply	             Source Over
svg:screen	    Screen	                 Source Over
svg:overlay	    Overlay	                 Source Over
svg:darken	    Darken	                 Source Over
svg:lighten	    Lighten	                 Source Over
svg:color-dodge	Color Dodge	             Source Over
svg:color-burn	Color Burn	             Source Over
svg:hard-light	Hard Light	             Source Over
svg:soft-light	Soft Light	             Source Over
svg:difference	Difference	             Source Over
svg:color	    Color	                 Source Over
svg:luminosity	Luminosity	             Source Over
svg:hue	        Hue	                     Source Over
svg:saturation	Saturation	             Source Over
svg:plus	    Normal	                 Lighter
svg:dst-in    	Normal	                 Destination In
svg:dst-out	    Normal	                 Destination Out
svg:src-atop	Normal	                 Source Atop
svg:dst-atop	Normal	                 Destination Atop
"""



"""
Modes in py blend

svg:src-over	*
svg:multiply	*
svg:screen	    Screen	                 Source Over
svg:overlay	    Overlay	                 Source Over
svg:darken	    *
svg:lighten	    *
svg:color-dodge	Color Dodge	             Source Over
svg:color-burn	Color Burn	             Source Over
svg:hard-light	*
svg:soft-light	*
svg:difference	*
svg:color	    Color	                 Source Over
svg:luminosity	Luminosity	             Source Over
svg:hue	        Hue	                     Source Over
svg:saturation	Saturation	             Source Over
svg:plus	    Normal	                 Lighter
svg:dst-in    	Normal	                 Destination In
svg:dst-out	    Normal	                 Destination Out
svg:src-atop	Normal	                 Source Atop
svg:dst-atop	Normal	                 Destination Atop


"""

from pyora.Blend import *
from pyora.BlendNonSep import *
from pyora.Composite import *

blend_modes = {'svg:multiply': multiply, 'svg:screen':screen,'svg:overlay':overlay, 'svg:darken':darken_only,
               'svg:lighten':lighten_only, 'svg:color-dodge':dodge, 'svg:color-burn':burn,
               'svg:hard-light':hard_light, 'svg:soft-light':soft_light, 'svg:difference':difference,
               'svg:color':color, 'svg:luminosity':luminosity, 'svg:hue':hue, 'svg:saturation':saturation,

               }
composite_modes = {'svg:plus': plus, 'svg:dst-in': dst_in, 'svg:dst-out': dst_out, 'svg:src-atop': src_atop,
                   'svg:dst-atop': dst_atop}

"""
The short documentation has this to say about compositing:

Isolated groups are always rendered independently at first, starting with a fully-transparent ‘black’ backdrop
 (rgba={0,0,0,0}). The results of this independent composite are then rendered on top of the group’s own backdrop 
 using the group’s opacity and composite mode settings. Conversely non-isolated groups are rendered by rendering 
 each child layer or sub-stack in turn to the group’s backdrop, just as if there were no stacked group.

The root stack has a fixed, implicit rendering in OpenRaster: it is to composite as an isolated group over a 
background of the application’s choice.

Non-root stacks should be rendered as isolated groups if: a) their isolation property is isolate (and not auto); 
or b) their opacity is less that 1.0; or c) they use a composite-op other than svg:src-over. This inferential 
behaviour is intended to provide backwards compatibility with apps which formerly didn’t care about group isolation.

Applications may assume that all stacks are isolated groups if that is all they support. If they do so, 
they must declare when writing OpenRaster files that their layer groups are isolated (isolation='isolate'). 
"""

class Renderer:
    """

    """

    def __init__(self, project):
        self.project = project

    def pil2np(self, image):
        return np.array(image).astype(np.float)

    def np2pil(self, arr):
        return Image.fromarray(np.uint8(np.around(arr, 0)))

    def _render_two(self, backdrop, layer_data, layer):

        if layer.composite_op in blend_modes:
            _tmp_layer_data = blend_modes[layer.composite_op](layer_data,
                                                              backdrop,
                                                              1.0, layer.offsets)
            canvas = normal(
                _tmp_layer_data,
                backdrop,
                layer.opacity, (0, 0))

        elif layer.composite_op in composite_modes:
            canvas = composite_modes[layer.composite_op](layer_data,
                                                                  backdrop,
                                                                  layer.opacity, layer.offsets)

            # canvas = normal(
            #     _tmp_layer_data,
            #     backdrop,
            #     layer.opacity, layer.offsets)

        else:
            # assume svg:src-over
            canvas = normal(layer_data,
                            backdrop,
                            layer.opacity, layer.offsets)

        return canvas


    def render_isolated(self, backdrop, layers):
        """
        Render the provided layers together and return the resulting canvas
        The actual rendering process will consist of deciding which groups of layers should be rendered together
        using stacking and isolation rules, and then doing the actual blending + compositing with this function

        INPUT LAYERS TO THIS FUNCTION SHOULD BE ORDERED FROM LOWEST (below) to HIGHEST (above)

        The composite op is performed on everything below the top layer - the top layer is not directly effected
        However, only the output of the composite is kept, so the original top layer is removed so the algorithm is like
        from the bottom up. composite with layer directly below to make new canvas, etc, etc.
        :return:
        """
        canvas = backdrop

        for i, layer in enumerate(layers):
            layer_data = self.pil2np(layer.get_image_data(raw=True).convert('RGBA'))
            canvas = self._render_two(canvas, layer_data, layer)

        return canvas

    def render(self):
        """
        for each layer (except the lowest one), apply blend mode to it, from
        it and the layer below it
        :return:
        """
        canvas = self.pil2np(Image.new('RGBA', self.project.dimensions))

        # strategy:
        # iterate layers (absolute z index) from the bottom of the tree up
        # gather them into a list until you reach one that is in another group
        # render that list either with a blank canvas (for isolated) or the current canvas
        # set that result as the new current canvas
        # start iterating layers again until there is another group change
        # etc

        all_layers = list(self.project.iter_layers)

        current_group = all_layers[0].parent
        current_layers = []


        for i, layer in enumerate(all_layers):
            if layer.hidden:
                continue

            if layer.parent != current_group or i == len(all_layers)-1:

                # for the very last layer, we need to include it even though the group condition is different
                if i == len(all_layers)-1:
                    current_layers.append(layer)

                # render previous group, start new one

                if current_group.visible:
                    # determine if isolation is appropriate
                    if current_group.isolated or current_group.opacity < 1.0 or current_group.composite_op != 'svr:src-over':
                        backdrop = self.pil2np(Image.new('RGBA', self.project.dimensions, (0, 0, 0, 0)))
                    else:
                        backdrop = canvas


                    # blend the group with the current canvas
                    canvas = self._render_two(canvas, self.render_isolated(backdrop, current_layers), current_group)


                current_layers = []

            current_group = layer.parent
            current_layers.append(layer)

        return self.np2pil(canvas)


def make_thumbnail(image):
    # warning: in place modification
    if image.size[0] > 256 or image.size[1] > 256:
        image.thumbnail((256, 256))
