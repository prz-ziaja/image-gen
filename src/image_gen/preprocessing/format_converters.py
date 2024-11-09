from skimage.color import rgb2hsv


def rgb_to_hsv_map_batches(row, image_key, result_name="hsv", keep_source=True):
    imgs = row[image_key]
    # Convention - channels_first
    row[result_name] = rgb2hsv(imgs, channel_axis=-3)
    if not keep_source:
        _ = row.pop(image_key)

    return row
