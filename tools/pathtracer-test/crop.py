#!/usr/bin/env python3
"""Crop / compare helper for path tracer outputs (uses the project's .venv PIL).

    crop.py in.png out.png x0 y0 x1 y1 [scale]      crop a box, optionally upscale
    crop.py --stats in.png x0 y0 x1 y1              mean / std of a box (8-bit sRGB)
    crop.py --diff a.png b.png out.png              |a - b| x 4
    crop.py --sheet out.png cols scale a.png b.png ... contact sheet (optionally a crop:
            --sheet out.png cols scale --box x0 y0 x1 y1 a.png ...)
"""
import sys
from PIL import Image, ImageChops
import numpy as np

a = sys.argv[1:]
if a[0] == '--stats':
    im = np.asarray(Image.open(a[1]).convert('RGB')).astype(float)
    x0, y0, x1, y1 = map(int, a[2:6])
    box = im[y0:y1, x0:x1].reshape(-1, 3)
    print('mean', box.mean(0).round(1), 'std', box.std(0).round(2))
elif a[0] == '--diff':
    x = Image.open(a[1]).convert('RGB')
    y = Image.open(a[2]).convert('RGB').resize(x.size)
    d = np.asarray(ImageChops.difference(x, y)).astype(float) * 4
    Image.fromarray(np.clip(d, 0, 255).astype(np.uint8)).save(a[3])
    print('mean abs diff', (d / 4).mean().round(2))
elif a[0] == '--sheet':
    out, cols, scale = a[1], int(a[2]), float(a[3])
    rest = a[4:]
    box = None
    if rest[0] == '--box':
        box = tuple(map(int, rest[1:5]))
        rest = rest[5:]
    ims = []
    for f in rest:
        im = Image.open(f).convert('RGB')
        if box:
            im = im.crop(box)
        ims.append(im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS))
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out)
else:
    im = Image.open(a[0])
    x0, y0, x1, y1 = map(int, a[2:6])
    c = im.crop((x0, y0, x1, y1))
    if len(a) > 6:
        s = float(a[6])
        c = c.resize((int(c.width * s), int(c.height * s)), Image.NEAREST)
    c.save(a[1])
