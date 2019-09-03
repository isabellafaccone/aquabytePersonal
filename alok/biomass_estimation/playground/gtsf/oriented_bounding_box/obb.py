from numpy import ndarray, array, asarray, dot, cross, cov, array, finfo, min as npmin, max as npmax
from numpy.linalg import eigh, norm
from sklearn.decomposition import PCA


########################################################################################################################
# adapted from: http://jamesgregson.blogspot.com/2011/03/latex-test.html
########################################################################################################################
class OBB:
    def __init__(self):
        self.rotation = None
        self.min = None
        self.max = None

    def transform(self, point):
        return dot(array(point), self.rotation)

    @property
    def centroid(self):
        return self.transform((self.min + self.max) / 2.0)

    @property
    def extents(self):
        return abs(self.transform((self.max - self.min) / 2.0))

    @property
    def points(self):
        return [
            # upper cap: ccw order in a right-hand system
            # rightmost, topmost, farthest
            self.transform((self.max[0], self.max[1], self.min[2])),
            # leftmost, topmost, farthest
            self.transform((self.min[0], self.max[1], self.min[2])),
            # leftmost, topmost, closest
            self.transform((self.min[0], self.max[1], self.max[2])),
            # rightmost, topmost, closest
            self.transform(self.max),
            # lower cap: cw order in a right-hand system
            # leftmost, bottommost, farthest
            self.transform(self.min),
            # rightmost, bottommost, farthest
            self.transform((self.max[0], self.min[1], self.min[2])),
            # rightmost, bottommost, closest
            self.transform((self.max[0], self.min[1], self.max[2])),
            # leftmost, bottommost, closest
            self.transform((self.min[0], self.min[1], self.max[2])),
        ]

    @classmethod
    def build_from_points(cls, points):
        if not isinstance(points, ndarray):
            points = array(points, dtype=float)
        assert points.shape[1] == 3, 'points have to have 3-elements'

        pca = PCA()
        pca.fit(points)
        eigen_vectors = pca.components_

        obb = OBB()

        def try_to_normalize(v):
            n = norm(v)
            if n < finfo(float).resolution:
                raise ZeroDivisionError
            return v / n

        r = try_to_normalize(eigen_vectors[:, 0])
        u = try_to_normalize(eigen_vectors[:, 1])
        f = try_to_normalize(eigen_vectors[:, 2])

        obb.rotation = array((r, u, f)).T

        # apply the rotation to all the position vectors of the array
        # TODO : this operation could be vectorized with tensordot
        p_primes = asarray([obb.rotation.dot(p) for p in points])
        obb.min = npmin(p_primes, axis=0)
        obb.max = npmax(p_primes, axis=0)

        return obb, eigen_vectors
        
        
