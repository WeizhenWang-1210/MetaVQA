def label_transfer(nusc_img_path, label_img_path):
    from PIL import Image
    src = Image.open(nusc_img_path)
    label = Image.open(label_img_path)
    label = label.resize((1600, 900))

    mask = Image.new('L', label.size, 0)  # Create a new grayscale image for the mask
    for x in range(label.width):
        for y in range(label.height):
            r, g, b = label.getpixel((x, y))
            if (r, g, b) != (0, 0, 0):  # Check if the pixel is not black
                mask.putpixel((x, y), 255)  # Set mask pixel to white (fully opaque)

    src.paste(label, mask=mask)
    src.save("test_transfer.png")