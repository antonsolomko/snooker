# -*- coding: utf-8 -*-
"""
@author: Anton Solomko

Module provides tools for detecting a snooker table on an image.
The main function is detect_table
"""

import cv2
import numpy as np
from projection import Projection, Projection3d, cyclic_pairs

PI = np.pi

# Polar coordinates of a line are the polar coordinates (t,r) of 
# the normal vector connecting the origin with a point on the line
# t is an angle, measured in radians, r is a magnitude

def combinations(elements, k=4):
    """Generate all combinations of k elements (order matters)"""
    if k <= 0:
        yield []
    else:
        for i in range(k-1, len(elements)):
            tail = elements[i]
            for head in combinations(elements[:i], k-1):
                yield head + [tail]

def line_translation(origin):
    """
    Factory. Return a translation function that changes 
    line's polar coordinates under coordinate system translation
    """
    a, b = origin
    dist = np.sqrt(a**2 + b**2)
    angle = np.arctan2(b, a)
    def translation(line):
        t0, r0 = line
        r = r0 - dist*np.cos(t0-angle)
        t = t0
        if r < 0:
            t, r = t+PI, -r
        return t, r
    return translation

def lines_intersection(line1, line2, origin=(0,0)):
    """
    Return the intersection of two lines given in polar form, 
    (None,angle) if the lines are parallel
    """
    t1, r1 = line1
    t2, r2 = line2
    x_origin, y_origin = origin
    if (t1-t2)%PI == 0:  # parallel lines
        return None, t1%PI
    else:
        x1, y1, d1 = r1*np.cos(t1), r1*np.sin(t1), r1**2
        x2, y2, d2 = r2*np.cos(t2), r2*np.sin(t2), r2**2
        x = (d1*y2 - d2*y1) / (x1*y2 - x2*y1) + x_origin
        y = (d1*x2 - d2*x1) / (y1*x2 - y2*x1) + y_origin
        return x, y

def get_corners(lines, origin=(0,0)):
    """
    Return a list of vertices for a polygon given by lines.
    Edges are assumed to be ordered
    """
    res = [lines_intersection(L1,L2,origin) for L1,L2 in cyclic_pairs(lines)]
    if all(p[0] is not None for p in res):
        return res
    else:
        return None

def horizon_slope(lines):
    """
    Assuming the quadrilateral given by four lines is a perspective projection
    of a table, return the (absolute value of the) slope of the horizon
    """
    assert len(lines) == 4
    x1, y1 = lines_intersection(lines[0], lines[2])
    x2, y2 = lines_intersection(lines[1], lines[3])
    if x1 is not None and x2 is not None:
        res = np.arctan2(y1-y2, x1-x2)
    elif x1 is not None:
        res = y2 - PI/2
    elif x2 is not None:
        res = y1 - PI/2
    else:
        res = y1-y2
    res %= PI
    return min(res, PI-res)

def lines_distance(line1, line2, factor=0.01):
    """Distance between polar coordinated of two lines.
    Magnitudes are rescaled by factor"""
    t1,r1 = line1[:2]
    t2,r2 = line2[:2]
    dt = abs(t1-t2)
    t = (t1+t2) / 2
    factor *= 1 + np.cos(t-PI/2)**2 + 0.5*np.cos(t-3*PI/2)
    dt = min(dt, 2*PI-dt)
    return abs(r1-r2)*factor + dt

def lines_groups(lines, threshold=0.05):
    """
    Divide lines into groups depending on the angle, gaps between groups being
    bigger than threshold
    """
    if not lines:
        return []
    lines.sort()
    clusters, cluster = [], []
    angle = lines[0][0]
    for t,r in lines:
        if t-angle > threshold:
            clusters.append(cluster)
            cluster = []
        cluster.append((t,r))
        angle = t
    clusters.append(cluster)
    # Glue groups close to 0=2*PI
    if len(clusters) > 1 and 2*PI + lines[0][0] - lines[-1][0] < 0.1:
        clusters[0] += clusters[-1]
        del clusters[-1]
    # Within each group sort by distance to the origin
    for cluster in clusters:
        cluster.sort(key=lambda line: line[1])
    # Sort groups by distance to the origin
    clusters.sort(key=lambda lines: lines[0][1])
    return clusters

def lines_groups2(clusters, threshold=2):
    subclusters = []
    for lines in clusters:
        subcluster = []
        r0 = lines[0][1]
        for t,r in lines:
            if r-r0 > threshold:
                subclusters.append(subcluster)
                subcluster = []
            subcluster.append((t,r))
            r0 = r
        subclusters.append(subcluster)
    return subclusters

# SLOW
def group_lines_subbundles(lines):
    """Replace groups of close lines with their averages"""
    clusters = [(t,r,1) for t,r in lines]
    while True:
        min_dist = 1000
        for line1 in clusters:
            for line2 in clusters:
                dist = lines_distance(line1, line2)
                if 0 < dist < min_dist:
                    min_dist = dist
                    minline1, minline2 = line1, line2
        if min_dist < 0.2:
            clusters.remove(minline1)
            clusters.remove(minline2)
            (t1, r1, w1), (t2, r2, w2) = minline1, minline2
            t, r, w = (t1*w1+t2*w2)/(w1+w2), (r1*w1+r2*w2)/(w1+w2), w1+w2
            clusters.append((t,r,w))
        else:
            break
    return [(t,r) for t,r,w in clusters]

def group_lines_bundles(lines):
    groups = lines_groups2(lines_groups(lines))
    clusters = []
    for group in groups:
        d = group[0][1]
        cluster = []
        for t,r in group:
            if r-d > 20:
                clusters.append(cluster)
                cluster = []
            cluster.append((t,r))
            d = r
        clusters.append(cluster)
    res = []
    for cluster in clusters:
        res += group_lines_subbundles(cluster)
    return res

# SLOW
def green_mask(img):
    return None
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for ir, mr in zip(img, mask):
        for c, (b,g,r) in enumerate(ir):
            if g>b and g>r:
                mr[c] = 1
    return mask

def mean_deviation(img, corners, green_mask=None):
    """Return mean color and standard deviation for a polygonal region"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int_(corners)], 1)
    if green_mask is not None:
        mask = cv2.bitwise_and(mask, green_mask)
    mean, stddev = np.int_(cv2.meanStdDev(img, mask=mask))
    return mean[:,0], stddev[:,0]

def select_clusters(img, clusters, origin):
    """Return 4 borderlines that bound the table"""
    if len(clusters) < 4:
        return clusters, None
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    in_frame = lambda x,y: 0<=x<img_hsv.shape[1] and 0<=y<img_hsv.shape[0]
    attempt = 1
    for quadruple in combinations(clusters, 4):
        quadruple.sort()
        borders = [lines[0] for lines in quadruple]
        corners = get_corners(borders, origin)
        slope = horizon_slope(borders)
        if corners and slope<0.3:
            # average color in the central area
            mean_color,stddev = mean_deviation(img_hsv,corners,green_mask(img))
            for i in range(4):
                for line in reversed(quadruple[i]):
                    borders[i] = line
                    corners = get_corners(borders, origin)
                    if not all(in_frame(*c) for c in corners):
                        continue
                    pr = Projection(corners, origin)
                    inner_strip, outer_strip = pr.border_neighbourhoods(i-1)
                    inner_color,inner_dev = mean_deviation(img_hsv,inner_strip)
                    outer_color,outer_dev = mean_deviation(img_hsv,outer_strip)
                    inner_diff = np.abs(inner_color-mean_color)
                    outer_diff = np.abs(outer_color-mean_color)
                    
                    if TEST:
                        imgt = img.copy()
                        for line in borders:
                            draw_line(imgt, line, origin)
                        draw_polygon(imgt, inner_strip)
                        draw_polygon(imgt, outer_strip)
                        cv2.imwrite('data/lines/steps/%s_%02d.jpg'
                                    %(filename.replace('.',''),attempt), imgt)
                        attempt += 1
                        print('in %s out %s ind %s outd %s'%
                              (inner_diff, outer_diff, inner_dev, outer_dev))
                        
                    if (inner_diff[0]<10 and inner_dev[0]<20 
                            and max(inner_diff)<100 
                            and (outer_diff[0]>10 or outer_dev[0]>20)):
                        if TEST:
                            print('v')
                        break
                else:  # loop finished without break, no i'th borderline found
                    break  # stop search
            else:  # search was not stopped, all borderlines found
                clusters = [[line for line in lines if line[1]<border[1]] 
                            for lines,border in zip(quadruple,borders)]
                return clusters, borders
    return clusters, None

# The main function
def detect_table(img):
    """Return the table corners coordinates if detected, None otherwise"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 25, 100, apertureSize=3)
    lines = cv2.HoughLines(img_edges, 1, PI/720, 250)
    lines = [line[0][::-1] for line in lines]
    if len(lines) > 500:
        return None  # too many lines
    origin = (img.shape[1]//2, img.shape[0]//2)
    translate = line_translation(origin)
    lines = [translate(line) for line in lines]
    lines = group_lines_bundles(lines)
    clusters = lines_groups(lines)
    clusters, borders = select_clusters(img, clusters, origin)
    if borders:
        return get_corners(borders, origin)
    else:
        return None

# Drawing tools
def draw_segment(img, segment, color=(255,255,255), width=1):
    x1, y1, x2, y2 = (int(c+0.5) for c in segment)
    cv2.line(img, (x1,y1), (x2,y2), color, width)

def draw_line(img, line, origin=(0,0), color=(255,255,255), width=1):
    t, r = line
    x_origin, y_origin = origin
    diam = max(img.shape)
    a, b = np.cos(t), np.sin(t)
    x0 = a*r + x_origin
    y0 = b*r + y_origin
    segment = (x0-diam*b, y0+diam*a, x0+diam*b, y0-diam*a)
    draw_segment(img, segment, color, width)

def draw_polygon(img, corners, color=(255,255,255), width=1):
    for (x1,y1), (x2,y2) in cyclic_pairs(corners):
        draw_segment(img, (x1,y1,x2,y2), color, width)


TEST = 1
if __name__ == '__main__':
    import os

    processed = [f for f in os.listdir("data/lines/") if '.' in f]
    #processed = []
    files = [f for f in os.listdir("data/") if '.' in f and f not in processed]
    files = ['0017.png']
    for filename in files:
        print(filename)
        img = cv2.imread('data/' + filename)
        corners = detect_table(img)
        if corners:
            origin = img.shape[1]//2, img.shape[0]//2
            projection = Projection3d(corners, origin)
            print(projection.precision)
            
            transform = projection.space_to_screen
            outer_corners = [transform(c) for c in projection.table_corners]
            draw_polygon(img, outer_corners)
            inner_corners = [transform(c) for c in projection.play_area_corners]
            draw_polygon(img, inner_corners, color=(200,200,200))
            
            cv2.imwrite('data/lines/'+filename, img)
        else:
            print('-')