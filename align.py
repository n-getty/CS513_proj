import argparse
import math
import requests
from PIL import Image, ImageDraw
from StringIO import StringIO
import pandas as pd
from collections import defaultdict
import numpy as np
from math import sin, cos, sqrt, atan2, radians

level = 23

min_lat = -85.05112878;
max_lat = 85.05112878;
min_long = -180;
max_long = 180;


def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue);


def get_tile(lat,long):
    lat = clip(lat, min_lat, max_lat)
    long = clip(long, min_long, max_long)

    sinLatitude = math.sin(lat * math.pi / 180)

    pixelX = int(((long + 180) / 360.0) * 256 * (2 ** level))

    pixelY = int((0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)) * 256 * 2 ** level)

    map_size = 256 << level

    pixelX = clip(pixelX, 0, map_size - 1)
    pixelY = clip(pixelY, 0, map_size - 1)

    tileX = int(math.floor(pixelX/256))

    tileY = int(math.floor(pixelY/256))

    rX = pixelX % 256
    rY = pixelY % 256

    return tileX, tileY, rX, rY


def get_quadkey(tileX, tileY):
    quadkey = ''
    for i in range(1, level + 1)[::-1]:
        digit = 0;
        mask = 1 << (i - 1);
        if tileX & mask:
            digit += 1
        if tileY & mask:
            digit += 2

        quadkey += str(digit);

    return quadkey


def between_keys(tile1, tile2):
    between = []
    key_tile_dict = {}
    if tile1[0] < tile2[0] and tile1[1] < tile2[1]:
        ne_tile, sw_tile = tile1, tile2
    elif tile1[0] > tile2[0] and tile1[1] > tile2[1]:
        sw_tile, ne_tile = tile1, tile2
    else:
        return [get_quadkey(tile1[0], tile1[1])]
    cur = ne_tile[:]
    while cur[1] <= sw_tile[1]:
        row = []
        while cur[0] <= sw_tile[0]:
            key = get_quadkey(cur[0], cur[1])
            row.append(key)
            key_tile_dict[(cur[0], cur[1])] = key
            cur[0] += 1
        between.append(row)
        cur[1] += 1
        cur[0] = ne_tile[0]

    return between, key_tile_dict


def get_imgs(quads):
    base_url = 'http://h0.ortho.tiles.virtualearth.net/tiles/h'
    end_url = '.jpeg?g=131'

    tiles = []
    i = 0
    key_img_dict = {}
    for row in quads:
        t = []
        for q in row:
            i += 1
            url = base_url + q + end_url
            tile = StringIO(requests.get(url).content)
            img = Image.open(tile)
            key_img_dict[q] = img
            t.append(img)
        tiles.append(t)

    return tiles, key_img_dict


def paint_cloud(key_img_dict, key_tile_dict, point_pixs):
    for ts, pixels in point_pixs.items():
        q = key_tile_dict[ts]
        im = key_img_dict[q]
        draw = ImageDraw.Draw(im)
        #draw.line(pixels, fill=128, width=1)
        draw.point(pixels, fill=128)


def pil_grid(images, rX1, rX2, rY1, rY2):
    width = len(images[0])
    pix_width = images[0][0].size[0]
    pix_height = images[0][0].size[1]
    shape = (pix_width * width, pix_height * len(images))
    im_grid = Image.new('RGB', shape, color='white')

    for i, row in enumerate(images):
        for j, im in enumerate(row):
            im_grid.paste(im, (pix_width * j, pix_height * i))


    # Crop tile grid by original box specifications
    #im_grid = im_grid.crop((rX1, rY1, im_grid.size[0] - (256-rX2), im_grid.size[1] - (256-rY2)))
    return im_grid


def med(pc, d):
    for x in range(0,len(pc)-d, d):
        med5_lat = np.median(pc.lat.loc[x:x+d])
        med5_long = np.median(pc.long.loc[x:x+d])
        pc.lat.loc[x] = med5_lat
        pc.long.loc[x] = med5_long
        for i in range(1,d):
            pc.drop(x + i, inplace=True)


def dist(lat1, lat2, long1, long2):
    R = 6373.0
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def thresh(pc):
    dists = []
    for x in range(len(pc) - 1):
        d = dist(pc.lat.loc[x], pc.lat.loc[x+1], pc.long.loc[x], pc.long.loc[x+1])
        dists.append(d)

    med = np.mean(dists)
    while x < len(pc) - 2:
        d = dist(pc.lat.loc[x], pc.lat.loc[x+1], pc.long.loc[x], pc.long.loc[x+1])
        if d > med:
            pc.drop(x+1, inplace=True)
            x=x+2
        else:
            x=x+1


def main():
    global level
    level = 19

    print "Reading Point Cloud Data"
    pc = pd.read_csv('final_project_point_cloud.fuse', delimiter=' ', header=0, names=['lat', 'long'], usecols=[0,1])

    #pc = pd.read_csv('pointcloud2.fuse', delimiter=' ', header=0, names=['lat', 'long'], usecols=[0, 1])

    lat1, long1 = pc.lat.max(), pc.long.min()
    lat2, long2 = pc.lat.min(), pc.long.max()

    point_pixs = defaultdict(list)

    #Smooth with median
    med(pc, 15)

    #Use distance threshold to remove erroneous points
    #thresh(pc)

    print "Calculating point cloud tiles"
    for i, p in pc.iterrows():
        tileX1, tileY1, rX1, rY1 = get_tile(p.lat, p.long)
        point_pixs[(tileX1, tileY1)].append((rX1, rY1))

    print "Getting input coordinate tiles and pixels"
    tileX1, tileY1, rX1, rY1 = get_tile(lat1, long1)
    tileX2, tileY2, rX2, rY2 = get_tile(lat2, long2)

    print "Generating keys between tiles"
    keys, key_tile_dict = between_keys([tileX1, tileY1], [tileX2, tileY2])

    print "Retrieving images for each key"
    imgs, key_img_dict = get_imgs(keys)

    paint_cloud(key_img_dict, key_tile_dict, point_pixs)

    print "Stitching images and cropping to fit box"
    stitched = pil_grid(imgs, rX1, rX2, rY1, rY2)

    print "Saving image"
    stitched.save('out_med15.png')


if __name__ == '__main__':
    main()