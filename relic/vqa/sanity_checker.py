from PIL import Image
import numpy as np
from vqa.vqagen.utils.geometric_utils import get_distance


def world_annotation_checker(filepath, colormappath, multiviewmask=None):
    import json
    with open(filepath, "r") as f:
        world_annotation = json.load(f)
    with open(colormappath, "r") as f:
        id2color = json.load(f)
    mapidpool = set()
    for id, color in id2color.items():
        mapidpool.add(id)
    if multiviewmask is not None:
        mask = Image.open(multiviewmask)
        mask = np.array(mask)
        assert len(mask.shape) == 3 and mask.shape[2] == 3
        flattened = mask.reshape(-1, mask.shape[2])
        observable_criteria = lambda r, g, b, c: not (r == 1 and g == 1 and b == 1) and not (
                r == 0 and g == 0 and b == 0) and (c > 128)
        unique_colors, counts = np.unique(flattened, axis=0,
                                          return_counts=True)  # apparently, already in int, so thats good to know
        unique_colors, counts = unique_colors.tolist(), counts.tolist()
        unique_colors = [(r, g, b) for (r, g, b), c in zip(unique_colors, counts) if observable_criteria(r, g, b, c)]
        unique_colors = [(round(r / 255, 5), round(g / 255, 5), round(b / 255, 5)) for r, g, b in unique_colors]
    observable_count = 0
    visibleidpool = set()
    for obj in world_annotation["objects"]:
        if obj["visible"]:
            visibleidpool.add(obj["id"])
            observable_count += 1
            assert len(obj["observing_camera"]) > 0, "No visible camera observed {} despite annotated observable in {}"\
                .format(obj["id"], filepath)
        assert get_distance(obj["pos"], world_annotation["ego"]["pos"]) < 50, \
            "Object {} is {}m away from ego".format(obj["id"], get_distance(obj["pos"], world_annotation["ego"]["pos"]))
    for id in visibleidpool:
        assert id in mapidpool, "Can't establish color mapping for object{}".format(id)
        if multiviewmask is not None:
            color = id2color[id]
            assert (round(color[0], 5), round(color[1], 5),
                    round(color[2], 5)) in unique_colors, "Color associated with annotated-visible object {} not found \
                    in any segmentation mask for at least 128 pixels.".format(id)
    assert len(id2color) >= observable_count, \
        "The id2color dictionary observed {} items more than annotated observable objects".format(
            len(id2color) - observable_count)


if __name__ == "__main__":
    world_annotation_checker("verification_multiview/0_30_69/0_30/world_0_30.json",
                             "verification_multiview/0_30_69/0_30/metainformation_0_30.json",
                             "verification_multiview/0_30_69/0_30/multiview_mask_0_30.png")
