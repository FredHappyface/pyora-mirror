from PIL import Image


from pyora.Blend import *
from pyora.BlendNonSep import *
from pyora.Composite import *
from pyora import TYPE_LAYER, TYPE_GROUP

blend_modes = {'svg:multiply': multiply, 'svg:screen':screen,'svg:overlay':overlay, 'svg:darken':darken_only,
               'svg:lighten':lighten_only, 'svg:color-dodge':dodge, 'svg:color-burn':burn,
               'svg:hard-light':hard_light, 'svg:soft-light':soft_light, 'svg:difference':difference,
               }

blend_modes_nonsep = {
                'svg:color':color, 'svg:luminosity':luminosity, 'svg:hue':hue, 'svg:saturation':saturation,
                }

composite_modes = {'svg:src-over':src_over, 'svg:plus': plus,
                   'svg:dst-in': dst_in, 'svg:dst-out': dst_out, 'svg:src-atop': src_atop,
                   'svg:dst-atop': dst_atop
                   }

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

    def __init__(self, project):
        self._project = project

    def pil2np(self, image):
        return np.array(image).astype(np.float)

    def np2pil(self, arr):
        return Image.fromarray(np.uint8(np.around(arr, 0)))

    def _render_two(self, backdrop, layer_data, offsets, opacity, composite_op='svg:src-over'):
        """
        merge two layers of data together
        this is run progressively to paint each layer in the stack
        console.log('render 2 called', layer.name, layer.offsets)

        if (composite_op in blend_modes) {
            return self._blend_modes[blend_modes[composite_op]]
            (backdrop, layer_canvas, opacity, offsets[0], offsets[1],
                layer_canvas.width, layer_canvas.height);

        }else if(composite_op in blend_modes_nonsep) {
            return self._blend_modes_nonsep[blend_modes_nonsep[composite_op]]
            (backdrop, layer_canvas, opacity, offsets[0], offsets[1],
                layer_canvas.width, layer_canvas.height);

        }else if(composite_op in composite_modes) {
            return self._composite_modes[composite_modes[composite_op]]
            (backdrop, layer_canvas, opacity, offsets[0], offsets[1],
                layer_canvas.width, layer_canvas.height);

        }else {
            //assume svg:src - over
            return self._composite_modes.src_over(backdrop, layer_canvas, opacity,
                offsets[0], offsets[1], layer_canvas.width, layer_canvas.height);
        }
        """

        if composite_op in blend_modes:
            blend_func = blend_modes[composite_op]

        elif composite_op in blend_modes_nonsep:
            blend_func = blend_modes_nonsep[composite_op]

        elif composite_op in composite_modes:
            blend_func = composite_modes[composite_op]

        else:
            # assume svg:src-over
            blend_func = src_over

        with np.errstate(invalid='ignore', divide='ignore'):
            res = outer_shell(backdrop, layer_data, opacity, offsets, blend_func)

        return res

    #
    # def render_isolated(self, backdrop, layers):
    #     """
    #     Render the provided layers together and return the resulting canvas
    #     The actual rendering process will consist of deciding which groups of layers should be rendered together
    #     using stacking and isolation rules, and then doing the actual blending + compositing with this function
    #
    #     INPUT LAYERS TO THIS FUNCTION SHOULD BE ORDERED FROM LOWEST (below) to HIGHEST (above)
    #
    #     The composite op is performed on everything below the top layer - the top layer is not directly effected
    #     However, only the output of the composite is kept, so the original top layer is removed so the algorithm is like
    #     from the bottom up. composite with layer directly below to make new canvas, etc, etc.
    #     :return:
    #     """
    #     canvas = backdrop
    #
    #     for i, layer in enumerate(layers):
    #         layer_data = self.pil2np(layer.get_image_data(raw=True).convert('RGBA'))
    #         canvas = self._render_two(canvas, layer_data, layer)
    #
    #     return canvas

    def render(self, root_group=None):
        """Perform the full project render over the current project

        Args:
            root_group (pyora.Group): optional, instead of starting with the outermost group of the project,
                render just the specified group and its children


        Returns:
            PIL.Image: The fully composited image
        """

        """
        for each layer (except the lowest one), apply blend mode to it, from
        it and the layer below it

        strat:

        iterate layers, the isolated stack contains canvases for isolated groups we enter. When we enter a new
         isolated stack on our way up the tree, we append a blank canvas to this array. Painting of all layers
         proceeds on the last canvas in this array, until we reach the top of the isolated group. At that time we pop
         the last from the isolated stack and composite (or blend) it with the new last isolated stack canvas.
         Non isolated groups don't get an isolated canvas, but we maintain a current multiplier for the opacity value
         of all the non-isolated groups we enter, so that we can apply it to all layers inside of the non-isolated group

         For example, Group changes between layers 1 and 2.
         - If it gets deeper, we need to look up each parent
         above the new layer up to the parent we had last time. For each parent, if it is isolated, push to the
         isolated stack.
         - If it is shallower, will only change depth by one by definition. If isolated, pop the stack as said above.

        """
        
        canvas = self.pil2np(Image.new('RGBA', self._project.dimensions))
        
        all_children = list(self._project.iter_tree) if root_group is None else root_group.iter_tree
        current_group = self._project._root_group if root_group is None else root_group

        isolated_stacks = [canvas]
        non_isolated_alpha = 1.0


        # the last child will always just be the root group, so we don't need this in our main painting loop
        all_children.pop()


        for i, child in enumerate(all_children):

            def add_depeper_stacks():
                nonlocal non_isolated_alpha
                tmp_check_group = child.parent
                while tmp_check_group is not current_group:
                    # iterate upward to check for isolated stacks to create
                    if tmp_check_group._renders_isolated:
                        backdrop = self.pil2np(Image.new('RGBA', self._project.dimensions))
                        isolated_stacks.append(backdrop)
                    else:
                        non_isolated_alpha *= tmp_check_group.opacity
                    tmp_check_group = tmp_check_group.parent

            if child.type is TYPE_GROUP:
                if len(list(child.children)) == 0:
                    # if we have an empty group we don't need to render anything for it, but we still might need to
                    # assign a number of isolated or non-isolated new stacks as if it was a new child
                    # so we follow the same algorithm as the group change below
                    add_depeper_stacks()
                    current_group = child.parent
                    continue

                # one level shallower, close the group
                if child._renders_isolated:
                    # closing an isolated group, blend of composite with the next shallower isolated group
                    to_merge = isolated_stacks.pop()
                    merge_onto = isolated_stacks[-1]
                    isolated_stacks[-1] = self._render_two(merge_onto, to_merge,
                                                                                   current_group.offsets,
                                                                                   current_group.opacity,
                                                                                   current_group.composite_op)
                else:
                    non_isolated_alpha *= (1 / current_group.opacity)

                current_group = child.parent
                continue

            if child.parent is not current_group:
                # group change (deeper)
                # one or more levels deeper, might need to create more isolated stacks
                add_depeper_stacks()
                current_group = child.parent

            # load the layer image and draw it onto a canvas

            if child.hidden_rendered:
                layer_canvas = self.pil2np(Image.new('RGBA', self._project.dimensions))
            else:
                layer_canvas = self.pil2np(child.get_image_data(raw=False))

            merge_onto = isolated_stacks[-1]
            isolated_stacks[-1] = self._render_two(merge_onto, layer_canvas, (0, 0,),
                                                                           child.opacity * non_isolated_alpha,
                                                                           child.composite_op)

        if(len(isolated_stacks) != 1):
            print("pyora warning: Incorrect number of post-rendering isolated stacks, something went wrong!")

        return self.np2pil(isolated_stacks[0])


def make_thumbnail(image):
    # warning: in place modification
    if image.size[0] > 256 or image.size[1] > 256:
        image.thumbnail((256, 256))
