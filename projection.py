# -*- coding: utf-8 -*-
"""
@author: Anton Solomko

Module provides tools for transforming screen coordinates to space coordinates
and vise versa.

In space coordinates, table surface is parallel to the plane z=0, 
z-axis is oriented downwards,
balls centers have z coordinate equal to 0,
thus table cushions lie in the plane z=-CUSHION_HEIGHT,
the center of the table has coordinates x=0, y=0;
short side is parallel to x axis, long one - to y axis.
Corners of the table are ordered as follows: x<0 y<0, x<0 y>0, x>0 y>0, x>0 y<0

Projections depend on the following parameters:
ratio: ratio of the sides of the table, approximately 2,
alpha: rotation of the table in the xy-plane,
(a1,a2,a3): translation vector, position of the camera relative to the 
    table center,
beta: camera angle relative to the vertical direction, Pi/2 < beta < 0,
gamma: horizon slope,
scale: table coordinates are perspectively projected to the plane z=scale.

All lengths are measured in mm.
"""

import numpy as np
import scipy.optimize
from sympy import Matrix
import cv2

PLAY_AREA_WIDTH = 1750
PLAY_AREA_HEIGHT = 3500
CUSHION_WIDTH = 60
BALL_DIAMETER = 52.5
CUSHION_HEIGHT = 0.135 * BALL_DIAMETER
PLAY_AREA_DIMS = (PLAY_AREA_WIDTH, PLAY_AREA_HEIGHT)
TABLE_WIDTH = PLAY_AREA_WIDTH + 2 * CUSHION_WIDTH
TABLE_HEIGHT = PLAY_AREA_HEIGHT + 2 * CUSHION_WIDTH
SIDES_RATIO = TABLE_HEIGHT / TABLE_WIDTH
TABLE_DIMS = (TABLE_WIDTH, TABLE_HEIGHT)

class Record(): pass

def cyclic_pairs(elements):
    """
    Generate all pairs (elements[i], elements[i+1]),
    including the last with the first element
    """
    try:
        yield from zip(elements, elements[1:])
        yield elements[-1], elements[0]
    except IndexError:
        pass

def round_result(func):
    """Decorator. Round all the values returned by func to integers"""
    return lambda *args,**kargs: np.int_(func(*args,**kargs) + 0.5)

def lines_intersection(line1, line2):
    """
    Intersection of two lines, each given by a pair of points.
    Return None if parallel
    """
    (x1,y1), (x2,y2) = line1
    (x3,y3), (x4,y4) = line2
    d = (x2-x1)*(y4-y3) - (x4-x3)*(y2-y1)
    if d != 0:
        x = ((x2*y1-x1*y2)*(x4-x3) - (x4*y3-x3*y4)*(x2-x1)) / d
        y = ((x2*y1-x1*y2)*(y4-y3) - (x4*y3-x3*y4)*(y2-y1)) / d
        return np.array((x,y))
    else:
        return None

def rotation_xy(angle, dim=4):
    """Return rotation matrix"""
    assert dim >= 2
    res = np.identity(dim, dtype=float)
    c = np.cos(angle)
    s = np.sin(angle)
    res[:2,:2] = ((c,-s), (s,c))
    return res

def rotation_yz(angle, dim=4):
    """Return rotation matrix"""
    assert dim >= 3
    res = np.identity(dim, dtype=float)
    c = np.cos(angle)
    s = np.sin(angle) 
    res[1:3,1:3] = ((c,-s), (s,c))
    return res

def translation(a):
    """Return 4x4 translation matrix"""
    assert len(a) == 3
    res = np.identity(4, dtype=float)
    res[:3,3] = a
    return res

def rescale(scale, dim=4):
    """Rescale the first two coordinates by scale. Return diagonal matrix"""
    assert dim >= 2
    res = np.identity(dim, dtype=float)
    res[0,0] = res[1,1] = scale
    return res

def stretch(ratio, dim=4):
    """Stretch by ratio in y direction. Return diagonal matrix"""
    assert dim >= 2
    res = np.identity(dim, dtype=float)
    res[1,1] = ratio
    return res

def projective_rotation(beta, scale, gamma):
    """Factory. Return projective 2d transformation"""
    rotation = np.matmul(rotation_yz(-beta,dim=3), rotation_xy(-gamma,dim=3))
    def res(p):
        v = np.array((p[0],p[1],scale))
        x, y, z = np.matmul(rotation, v)
        return np.array((x,y)) / z
    return res

def rectangleness(corners):
    """
    Factory. Return two functions:
    error: for a vector of parameters (b,s,g) perform 
        a projective rotation of the corners and return deviation of the images
        from a rectangular shape (functional to be minimized);
    rescale: rescales parameters (b,s,g) to camera parameters
        (beta,scale,gamma)
    """
    max_y = -min(c[1] for c in corners)
    def rescale(b, s, g=0):
        scale = np.exp(s)
        beta = (b * (np.arctan2(max_y,scale) - np.pi/2)) % np.pi - np.pi
        gamma = 0.1 * g
        return beta, scale, gamma
    def error(args):
        """
        Return sum of |cos| over all four angles plus sides ratio deviation
        for the rotated shape
        """
        rotation = projective_rotation(*rescale(*args))
        rotated_corners = [rotation(c) for c in corners]
        sides = [p2-p1 for p1,p2 in cyclic_pairs(rotated_corners)]
        lens = [np.linalg.norm(s) for s in sides]
        directions = [s/t for s,t in zip(sides,lens)]
        ratio = (lens[0]+lens[2]) / (lens[1]+lens[3])
        if ratio < 1:
            ratio = 1 / ratio
        res = sum(abs(np.dot(d1,d2)) for d1,d2 in cyclic_pairs(directions))
        return res + (ratio-SIDES_RATIO)**2
    return error, rescale

# delete later
def plot_error(rect):
    N = 300
    img = np.zeros([N,N,3], dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            b = i/N
            g = 0
            s = 5 + 6*j/N
            v = rect([b,s,g])
            img[i][j] = [min(int(v*1000), 255)]*3
    cv2.imwrite('data/lines/min/2.jpg', img)

def rectangulation(corners):
    """
    Return (beta, scale, gamma=0) camera parameters for which 
    projective rotation maps corners to a rectangle like shape
    """
    error, rescale = rectangleness(corners)
    #plot_error(rect)
    res = scipy.optimize.minimize(error, x0=(0.5, 8), method='Powell')
    return rescale(*(res.x))

def rectangulate_reorder(screen_coordinates):
    """
    Reorder screen_coordinates to match the canonical corners order.
    Return a record containing attributes:
    beta, scale, gamma: rectangulation parameters,
    screen_coordinates: reordered screen coordinates, 
    corners: screen_coordinates images under projective rotation, reordered,
    sides: sides lengths,
    ratio: longer sides to shorter sides ratio,
    order: corners reordering
    """
    res = Record()
    res.beta, res.scale, res.gamma = rectangulation(screen_coordinates)
    rotation = projective_rotation(res.beta, res.scale, res.gamma)
    res.corners = [rotation(c) for c in screen_coordinates]
    res.order = [0,1,2,3]
    res.screen_coordinates = screen_coordinates
    directions = [p2-p1 for p1,p2 in cyclic_pairs(res.corners)]
    if any(x1*y2 > x2*y1 for (x1,y1),(x2,y2) in cyclic_pairs(directions)):
        for seq in res.screen_coordinates, res.corners, res.order:
            seq[:] = seq[::-1]
    res.sides = [np.linalg.norm(p2-p1) for p1,p2 in cyclic_pairs(res.corners)]
    res.ratio = (res.sides[0]+res.sides[2]) / (res.sides[1]+res.sides[3])
    if res.ratio < 1:
        for seq in res.screen_coordinates, res.corners, res.order, res.sides:
            seq[:] = seq[1:] + [seq[0]]
        res.ratio = 1 / res.ratio
    return res

def projection(ratio, alpha, beta, gamma, a1, a2, a3, scale, origin=(0,0)):
    """
    Factory. Returned function performs space to screen projection as follows:
    srtetch in y directin by ratio,
    rotate around z by alpha,
    translate by vector (a1,a2,a3),
    rotate around x by beta,
    rotate around z by gamma,
    project to the plane z=scale
    translate obtained 2D coordinates by origin
    """
    transform = stretch(ratio)
    transform = np.matmul(rotation_xy(alpha), transform)
    transform = np.matmul(translation((a1, a2, a3)), transform)
    transform = np.matmul(rotation_yz(beta), transform)
    transform = np.matmul(rotation_xy(gamma), transform)
    transform = np.matmul(rescale(scale), transform)
    def res(space_coordinates):
        v = np.array((0,0,0,1))
        v[:len(space_coordinates)] = space_coordinates
        x, y, z, _ = np.matmul(transform, v)
        return np.array((x,y)) / z + origin
    return res

def target(screen_coordinates, z=0):
    """
    Factory. Returned function projects canonical (square table) coorditates
    and returns the images difference with the screen_coordinates,
    functional to be minimized
    """
    assert len(screen_coordinates) == 4
    w = TABLE_WIDTH / 2
    corners = ((-w,-w,z),(-w,w,z),(w,w,z),(w,-w,z))
    def error(args):
        transform = projection(*args)
        corners_projections = np.array([transform(c) for c in corners])
        return np.linalg.norm(corners_projections - screen_coordinates)
    return error

def fit(screen_coordinates, z=0):
    """
    Return optimal camera parameters that minimize the difference between 
    canonical (square) corners images under projection and screen_coordinates
    """
    r = rectangulate_reorder(screen_coordinates)
    x, y = r.corners[2]-r.corners[1]
    alpha = np.arctan2(y, x)
    a3 = 2 * TABLE_WIDTH / (r.sides[1]+r.sides[3])
    table_center = lines_intersection(
            (r.corners[0],r.corners[2]), (r.corners[1],r.corners[3]))
    a1, a2 = table_center * a3
    solution = scipy.optimize.minimize(
            target(r.screen_coordinates, z), 
            x0 = (r.ratio, alpha, r.beta, r.gamma, a1, a2, a3-z, r.scale),
            method='Powell')
    solution.order = r.order
    return solution

#The following two functions provide a light version of projective 
#transformation that simply maps given four 2d points to the other four ones
def projection_matrix(coord_from, coord_to):
    equation_matrix = []
    for (xs, ys), (x, y) in zip(coord_from, coord_to):
        equation_matrix.append([xs, ys, 1, 0, 0, 0, -x*xs, -x*ys, -x])
        equation_matrix.append([0, 0, 0, xs, ys, 1, -y*xs, -y*ys, -y])
    kernel = Matrix(equation_matrix).nullspace()[0]
    return Matrix([kernel[0:3], kernel[3:6], kernel[6:9]])

def projection2d(coordinates_from, coordinates_to):
    """
    Factory. Return a projective transformation that maps 
    coordinates_from to coordinates_to
    """
    assert len(coordinates_from) == 4 and len(coordinates_to) == 4
    transform = projection_matrix(coordinates_from, coordinates_to)
    def res(point):
        x0, y0 = point
        x, y, z = transform * Matrix([[x0],[y0],[1]])
        return np.array((x,y)) / z
    return res


class Projection():
    def __init__(self, screen_coordinates, screen_center=(0,0),
                 table_dimensions=TABLE_DIMS):
        """
        screen_coordinates
        """
        assert len(screen_coordinates) == 4
        self.table_dimensions = table_dimensions
        self.origin = screen_center
        self.set_measures()
        coordinates = [np.array(c)-self.origin for c in screen_coordinates]
        r = rectangulate_reorder(coordinates)
        self.order = r.order
        self.ratio = r.ratio
        coordinates = [screen_coordinates[i] for i in self.order]
        self.space_to_screen = projection2d(self.table_corners, coordinates)
        self.screen_to_space = projection2d(coordinates, self.table_corners)
    
    def set_measures(self):
        self.table_dimensions = np.array(self.table_dimensions)
        self.play_area_dimensions = self.table_dimensions - 2*CUSHION_WIDTH
        corners_order = ((-1,-1), (-1,1), (1,1), (1,-1))
        corner = self.play_area_dimensions / 2
        self.play_area_corners = [corner*s for s in corners_order]
        corner = self.table_dimensions / 2
        self.table_corners = [corner*s for s in corners_order]
        corner += 1.5*CUSHION_WIDTH
        self.outer_corners = [corner*s for s in corners_order]
        w = CUSHION_WIDTH
        shifts = [(0,w), (w,0), (0,-w), (-w,0)]
        self.cushions = [[self.play_area_corners[i] + s, 
                          self.play_area_corners[(i+1)%4] - s,
                          self.table_corners[(i+1)%4] - s, 
                          self.table_corners[i] + s] 
                         for i,s in enumerate(shifts)]
        
    
    def border_neighbourhoods(self, i):
        """
        Return strips along each side of the i'th border 
        (connecting corners i and i+1)
        """
        j0 = self.order.index(i%4)
        j1 = self.order.index((i+1)%4)
        j = j0 if (j1-j0)%4 == 1 else j1
        inner = [self.space_to_screen(c) for c in self.cushions[j]]
        outer = [self.outer_corners[j0], self.outer_corners[j1],
                 self.table_corners[j1], self.table_corners[j0]]
        outer = [self.space_to_screen(c) for c in outer]
        return inner, outer
    
    def __repr__(self):
        res = '%s\n' % self.__class__.__name__
        for k, v in sorted(self.__dict__.items()):
            res += '  %s:  %s\n' % (k,v)
        return res


class Projection3d(Projection):
    """
    For given four screen coordinates of table corners computes camera 
    angles, distance to the table etc. and reconstructs projection map.
    Attributes space_to_screen and screen_to_space are functions that
    transform space coordinates to screen coordinates and vise versa.
    """
    def __init__(self, screen_coordinates, screen_center=(0,0),
                 table_dimensions=TABLE_DIMS):
        assert len(screen_coordinates) == 4
        self.table_dimensions = table_dimensions
        self.origin = screen_center
        self.set_measures()
        coordinates = [np.array(c)-self.origin for c in screen_coordinates]
        f = fit(coordinates, -CUSHION_HEIGHT)
        self.order = f.order
        self.precision = f.fun
        ratio, self.alpha, self.beta, self.gamma, *a, self.scale = f.x
        self.ratio = ratio*self.table_dimensions[0]/self.table_dimensions[1]
        self.a = np.array(a)
        self.space_to_screen = self.__space_to_screen()
        self.screen_to_space = self.__screen_to_space()
    
    def set_measures(self):
        Projection.set_measures(self)
        z = -CUSHION_HEIGHT
        for seq in (self.play_area_corners, self.table_corners, 
                    self.outer_corners):
            seq[:] = [np.append(xy,z) for xy in seq]
    
    def __space_to_screen(self):
        return projection(self.ratio, self.alpha, self.beta, self.gamma, 
                          *self.a, self.scale, self.origin)
        
    def __screen_to_space(self):
        pre_rotation = projective_rotation(self.beta, self.gamma, self.scale)
        post_rotation = np.matmul(stretch(1/self.ratio, dim=2),
                                  rotation_xy(-self.alpha, dim=2))
        *axy, az = self.a
        def res(screen_coordinates, z=0):
            p = pre_rotation(screen_coordinates) * (az+z) - axy
            return np.matmul(post_rotation, p)
        return res


if __name__ == '__main__':
    corners = [(472,136), (-640,-44), (14,-417), (647,-371)]
    p = Projection3d(corners)
    print(p)