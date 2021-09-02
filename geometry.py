
import numpy as np
T = np.array([[0, -1], [1, 0]])

# @Reference https://blog.csdn.net/daming98/article/details/79561777
# @Reference https://segmentfault.com/a/1190000004457595?ref=myread
def HasIntersection(A, B, C, D, Threshold=1.0e-9): 
    # @param A, B, C, D np.ndarray [PointNum, (x, y)]
    # Judge whether segment AB intersects CD.
    AC = C - A
    AD = D - A
    BC = C - B
    BD = D - B
    return (np.cross(AC, AD) * np.cross(BC, BD) <= Threshold) * (np.cross(AC, BC) * np.cross(AD, BD) <= Threshold)

def InterceptRatio(p1, p2, ref1, ref2, p1p2=None):
    # Judge whether p1p2 will be intercepted by ref1-ref2.
    # If yes, gives ratio: p1-InterceptionPoint / ref1-ref2
    # If no, gives ratio 1.0.
    PointNum = p1.shape[0]
    Lambda = np.ones((PointNum), )
    hasIntersection = HasIntersection(p1, p2, ref1, ref2)
    hasIntersectionIndices = np.argwhere(hasIntersection)
    p1WithIntersection = p1[hasIntersectionIndices, :]
    intersectionPoins = IntersectionPoints(p1WithIntersection, p2[hasIntersectionIndices, :], ref1, ref2)
    if p1p2 is None:
        p1p2 = p2 - p1
    LambdaWithIntersection = np.mean((intersectionPoins - p1WithIntersection) / p1p2, axis=1)
    Lambda[hasIntersectionIndices] = LambdaWithIntersection
    return Lambda

def IntersectionPoint(A, B, C, D):
    return
    
def HasIntersection_(P1, P2, Q1, Q2):
    isBoundaryBoxIntersect = IsBoundaryBoxIntersect(P1, P2, Q1, Q2)    
    return 

def IsBoundaryBoxIntersect(P1, P2, Q1, Q2): # 快速排斥实验
    P1X, P1Y, P2X, P2Y = P1[:, 0], P1[:, 1], P2[:, 0], P2[:, 1],
    Q1X, Q1Y, Q2X, Q2Y = Q1[:, 0], Q1[:, 1], Q2[:, 0], Q2[:, 1],
    return \
    np.min(P1X, P2X, axis=1) <= np.max(Q1X, Q2X, axis=1) * \
    np.min(Q1X, Q2X, axis=1) <= np.max(P1X, P2X, axis=1) * \
    np.min(P1Y, P2Y, axis=1) <= np.max(Q1Y, Q2Y, axis=1) * \
    np.min(Q1Y, Q2Y, axis=1) <= np.max(P1Y, P2Y, axis=1)
    # return \
    # np.min(P1[:, 0], P2[:, 0], axis=1) <= np.max(Q1[:, 0], Q2[:, 0], axis=1) * \
    # np.min(Q1[:, 0], Q2[:, 0], axis=1) <= np.max(P1[:, 0], P2[:, 0], axis=1) * \
    # np.min(P1[:, 1], P2[:, 1], axis=1) <= np.max(Q1[:, 1], Q2[:, 1], axis=1) * \
    # np.min(Q1[:, 1], Q2[:, 1], axis=1) <= np.max(P1[:, 1], P2[:, 1], axis=1)

def IsLineSegmentCross(P1, P2, Q1, Q2): # 跨立实验
    if \
        ((Q1[:, 0]-P1[:, 0])*(Q1[:, 1]-Q2[:, 1])-(Q1[:, 1]-P1[:, 1])*( Q1[:, 0]-Q2[:, 0])) * ((Q1[:, 0]-P2[:, 0])*(Q1[:, 1]-Q2[:, 1])-(Q1[:, 1]-P2[:, 1])*(Q1[:, 0]-Q2[:, 0])) < 0 or \
        ((P1[:, 0]-Q1[:, 0])*(P1[:, 1]-P2[:, 1])-(P1[:, 1]-Q1[:, 1])*(P1[:, 0]-P2[:, 0])) * ((P1[:, 0]-Q2[:, 0])*(P1[:, 1]-P2[:, 1])-(P1[:, 1]-Q2[:, 1])*( P1[:, 0]-P2[:, 0])) < 0:
        return True
    else:
       return False


# #
# # line segment intersection using vectors
# # see Computer Graphics by F.S. Hill
# #
# from numpy import *
# def perp( a ) :
#     b = empty_like(a)
#     b[0] = -a[1]
#     b[1] = a[0]
#     return b

# # line segment a given by endpoints a1, a2
# # line segment b given by endpoints b1, b2
# # return 
# def seg_intersect(a1,a2, b1,b2) :
#     da = a2-a1
#     db = b2-b1
#     dp = a1-b1
#     dap = perp(da)
#     denom = dot( dap, db)
#     num = dot( dap, dp )
#     return (num / denom.astype(float))*db + b1

def _Perpendicular(a):
    b = np.empty_like(a)
    b[:, 0] = -a[:, 1]
    b[:, 1] = a[:,0]
    return b

def IntersectionPoints(P1, P2, Q1,Q2) :
    P1P2 = P2 - P1
    Q1Q2 = Q2 - Q1
    Q1P1 = P1 - Q1
    P1P2Perpendicular = _Perpendicular(P1P2)
    Denominator = np.sum(P1P2Perpendicular * Q1Q2, axis=1)
    Numerator = np.sum(P1P2Perpendicular * Q1P1, axis=1)
    return (Numerator / Denominator.astype(float)) * Q1Q2 + Q1

def main():



    return


if __name__=="__main__":
    main()
