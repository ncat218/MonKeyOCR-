"""Fixed overlapping 2x2 geometry and polygon NMS; no GT and no pixel restoration."""
import math
import numpy as np
from shapely.geometry import Polygon


def tile_boxes(width,height,overlap=.2):
    if width<1 or height<1 or not 0<=overlap<1:raise ValueError('Invalid image dimensions/overlap')
    tw=min(width,int(math.ceil(width/(2-overlap))))
    th=min(height,int(math.ceil(height/(2-overlap))))
    return sorted(set((x,y,x+tw,y+th) for y in (0,height-th) for x in (0,width-tw)))


def translate_polygon(coordinates,x,y):
    pts=np.asarray(coordinates,dtype=float).reshape(-1,2)+[x,y]
    return pts.reshape(-1).tolist()


def seam_eligible(coords,tile,image_size,margin):
    x0,y0,x1,y1=tile;w,h=image_size
    pts=np.asarray(coords,dtype=float).reshape(-1,2)
    lo=pts.min(axis=0);hi=pts.max(axis=0)
    return not ((x0>0 and lo[0]<=x0+margin) or (y0>0 and lo[1]<=y0+margin) or
                (x1<w and hi[0]>=x1-margin) or (y1<h and hi[1]>=y1-margin))


def merge_views(views,iou_threshold):
    if not 0<iou_threshold<=1:raise ValueError('NMS IoU must be in (0,1]')
    candidates=[]
    for view in views:
        assert len(view['polygons'])==len(view['scores'])==len(view['eligible'])
        for poly,score,eligible in zip(view['polygons'],view['scores'],view['eligible']):
            if eligible:candidates.append((poly,float(score)))
    order=sorted(range(len(candidates)),key=lambda i:(-candidates[i][1],i))
    shapes=[]
    for poly,_ in candidates:
        geom=Polygon(np.asarray(poly,dtype=float).reshape(-1,2))
        if not geom.is_valid:geom=geom.buffer(0)
        shapes.append(geom)
    selected=[]
    for i in order:
        geom=shapes[i]
        # Keep degenerate outputs so the common evaluator can count them as FP.
        if geom.is_empty or geom.area<=0:
            selected.append(i);continue
        suppress=False
        for j in selected:
            other=shapes[j]
            if other.is_empty or other.area<=0:continue
            if geom.bounds[2]<other.bounds[0] or other.bounds[2]<geom.bounds[0] or geom.bounds[3]<other.bounds[1] or other.bounds[3]<geom.bounds[1]:continue
            intersection=geom.intersection(other).area
            union=geom.area+other.area-intersection
            if union>0 and intersection/union>iou_threshold:
                suppress=True;break
        if not suppress:selected.append(i)
    return dict(polygons=[candidates[i][0] for i in selected],scores=[candidates[i][1] for i in selected],
                candidates_before_nms=len(candidates),suppressed=len(candidates)-len(selected))
