import numba as nb
import math


@nb.jit(nopython=True, fastmath=True)
def dot(a, b, n, ofa=0, ofb=0):
    # Dotting a and b
    v = 0
    for i in range(n):
        v += a[ofa + i]*b[ofb + i]

    return v


@nb.jit(nopython=True)
def dist2d_at(a, b, p):
    # Vector from a to b
    abx = b[0] - a[0]
    aby = b[1] - a[1]

    # Vector from a to p
    apx = p[0] - a[0]
    apy = p[1] - a[1]

    # dot(ab ,ap) / dot(ab, ab)
    t = (abx*apx + aby*apy) / (abx**2 + aby**2)

    if t < 0 or t > 1:
        t = max(0.0, min(1.0, t))
        status = 1
    else:
        status = 0
    
    # proj(=a + t*ab) - p
    hx = a[0] + t*abx - p[0]
    hy = a[1] + t*aby - p[1]
    
    # Distance
    return status, math.sqrt(hx**2 + hy**2)


@nb.jit(nopython=True)
def dist3d_line_at(a, b, p):
    # Vector from a to b
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    abz = b[2] - a[2]

    # Vector from a to p
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    apz = p[2] - a[2]

    # dot(ab ,ap) / dot(ab, ab)
    t = (abx*apx + aby*apy + abz*apz) / (abx**2 + aby**2 + abz**2)
    t = max(0.0, min(1.0, t))
    
    # proj(=a + t*ab) - p
    hx = a[0] + t*abx - p[0]
    hy = a[1] + t*aby - p[1]
    hz = a[2] + t*abz - p[2]
    
    # Distance
    return math.sqrt(hx**2 + hy**2 + hz**2)


@nb.jit(nopython=True)
def dist3d_at(v0, v1, v2, p):
    # Vectors for the first edge (0, 1)
    e0x = v1[0] - v0[0]
    e0y = v1[1] - v0[1]
    e0z = v1[2] - v0[2]

    # Vectors for the first edge (0, 2)
    e1x = v2[0] - v0[0]
    e1y = v2[1] - v0[1]
    e1z = v2[2] - v0[2]

    # Vectors from p to v0
    pv0x = v0[0] - p[0]
    pv0y = v0[1] - p[1]
    pv0z = v0[2] - p[2]

    # Inner product of vectors
    a = e0x*e0x + e0y*e0y + e0z*e0z
    b = e0x*e1x + e0y*e1y + e0z*e1z
    c = e1x*e1x + e1y*e1y + e1z*e1z
    d = e0x*pv0x + e0y*pv0y + e0z*pv0z
    e = e1x*pv0x + e1y*pv0y + e1z*pv0z

    # Based on David Eberly's geometric tools
    det = a*c - b**2
    s = b*e - c*d
    t = b*d - a*e

    if det < 0:
        print(p)

    status = 1
    if s < 0:
        return status, dist3d_line_at(v0, v2, p)
    elif t <0:
        return status, dist3d_line_at(v0, v1, p)
    elif s + t > det:
        return status, dist3d_line_at(v1, v2, p)
    else:
        #if (s + t <=det) and (s >= 0) and (t >=0):
        # Inside of region, find a local coordinate
        inv_det = 1.0/det
        s *= inv_det
        t *= inv_det

        hx = v0[0] + s*e0x + t*e1x - p[0]
        hy = v0[1] + s*e0y + t*e1y - p[1]
        hz = v0[2] + s*e0z + t*e1z - p[2]

        # Noraml vector (outward)
        nx = e0y*e1z - e0z*e1y
        ny = e0z*e1x - e0x*e1z
        nz = e0x*e1y - e0y*e1x

        # Check outward or not
        if nx*hx + ny*hy + nz*hz < 0:
            status = 0

        return status, math.sqrt(hx**2+ hy**2 + hz**2)
