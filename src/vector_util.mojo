# geometry-central utilities: vector2.h, vector3.h, elementary_geometry.h
# Vector2 doubles as a complex number, exactly as upstream: `a * b` and `a / b` between two
# Vector2 values are complex multiply/divide, not dot/cross.

from std.math import sqrt, atan2, cos, sin, fabs


struct Vector2:
    var x: Float64
    var y: Float64

    def __init__(out self, x: Float64 = 0.0, y: Float64 = 0.0):
        self.x = x^
        self.y = y^

    def __add__(self, b: Self) -> Self:
        return Vector2(self.x + b.x, self.y + b.y)

    def __sub__(self, b: Self) -> Self:
        return Vector2(self.x - b.x, self.y - b.y)

    def __neg__(self) -> Self:
        return Vector2(-self.x, -self.y)

    def __mul__(self, b: Self) -> Self:
        return Vector2(self.x * b.x - self.y * b.y, self.x * b.y + self.y * b.x)

    def __mul__(self, s: Float64) -> Self:
        return Vector2(self.x * s, self.y * s)

    def __truediv__(self, b: Self) -> Self:
        var d = b.norm2()
        if d == 0.0:
            return Vector2(0.0, 0.0)
        return Vector2((self.x * b.x + self.y * b.y) / d, (self.y * b.x - self.x * b.y) / d)

    def __truediv__(self, s: Float64) -> Self:
        return Vector2(self.x / s, self.y / s)

    def norm2(self) -> Float64:
        return self.x * self.x + self.y * self.y

    def norm(self) -> Float64:
        return sqrt(self.norm2())

    def unit(self) -> Self:
        if self.norm2() == 0.0:
            return Vector2(0.0, 0.0)
        return self / self.norm()

    def copy(self) -> Self:
        return Vector2(self.x, self.y)

    def rotate90(self) -> Self:
        return Vector2(-self.y, self.x)

    def dot(self, b: Self) -> Float64:
        return self.x * b.x + self.y * b.y

    def conj(self) -> Self:
        return Vector2(self.x, -self.y)

    def inv(self) -> Self:
        return self / self.norm2()

    def arg(self) -> Float64:
        return atan2(self.y, self.x)

    def __getitem__(self, i: Int) -> Float64:
        if i == 0:
            return self.x
        return self.y


def v2_from_angle(theta: Float64) -> Vector2:
    return Vector2(cos(theta), sin(theta))


def v2_normalize(v: Vector2) -> Vector2:
    return v.unit()


def v2_normalize_cutoff(v: Vector2, mag: Float64 = 0.0) -> Vector2:
    if v.norm2() > mag * mag:
        return v.unit()
    return v.copy()


def v2_dot(a: Vector2, b: Vector2) -> Float64:
    return a.x * b.x + a.y * b.y


def v2_cross(a: Vector2, b: Vector2) -> Float64:
    return a.x * b.y - a.y * b.x


def nan64() -> Float64:
    return Float64(0.0) / Float64(0.0)


def v2_undefined() -> Vector2:
    return Vector2(nan64(), nan64())


def v2_is_defined(v: Vector2) -> Bool:
    return not (isnan(v.x) or isnan(v.y))


struct Vector3:
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64 = 0.0, y: Float64 = 0.0, z: Float64 = 0.0):
        self.x = x^
        self.y = y^
        self.z = z^

    def __add__(self, b: Self) -> Self:
        return Vector3(self.x + b.x, self.y + b.y, self.z + b.z)

    def __sub__(self, b: Self) -> Self:
        return Vector3(self.x - b.x, self.y - b.y, self.z - b.z)

    def __neg__(self) -> Self:
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, s: Float64) -> Self:
        return Vector3(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s: Float64) -> Self:
        return Vector3(self.x / s, self.y / s, self.z / s)

    def norm2(self) -> Float64:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> Float64:
        return sqrt(self.norm2())

    def unit(self) -> Self:
        if self.norm2() == 0.0:
            return Vector3(0.0, 0.0, 0.0)
        return self / self.norm()

    def copy(self) -> Self:
        return Vector3(self.x, self.y, self.z)

    def removeComponent(self, n: Self) -> Self:
        return self - n * v3_dot(self, n)

    def rotateAround(self, axis: Self, angle: Float64) -> Self:
        var c = cos(angle)
        var s = sin(angle)
        return self * c + v3_cross(axis, self) * s + axis * (v3_dot(self, axis) * (1.0 - c))

    # Vector3::buildTangentBasis
    def buildTangentBasis(self) -> (Self, Self):
        if fabs(self.x) <= fabs(self.y) and fabs(self.x) <= fabs(self.z):
            var v1 = Vector3(0.0, -self.z, self.y)
            return (v1^, v3_cross(self, v1))
        elif fabs(self.y) <= fabs(self.z):
            var v1 = Vector3(-self.z, 0.0, self.x)
            return (v1^, v3_cross(self, v1))
        else:
            var v1 = Vector3(-self.y, self.x, 0.0)
            return (v1^, v3_cross(self, v1))


def v3_dot(a: Vector3, b: Vector3) -> Float64:
    return a.x * b.x + a.y * b.y + a.z * b.z


def v3_cross(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def v3_unit(v: Vector3) -> Vector3:
    return v.unit()


def v3_normalize_cutoff(v: Vector3, mag: Float64 = 0.0) -> Vector3:
    if v.norm2() > mag * mag:
        return v.unit()
    return v


# Vector3.ipp: angleInPlane
def angle_in_plane(u: Vector3, v: Vector3, normal: Vector3) -> Float64:
    var n = normal.unit()
    var u_plane = (u - n * v3_dot(u, n)).unit()

def _det3(
    a: Float64, b: Float64, c: Float64, d: Float64, e: Float64, f: Float64, g: Float64,
    h: Float64, i: Float64,
) -> Float64:
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# vector_util.mojo: elementary_geometry.cpp: inCircleTest. This is the reference transcription of the
# 4x4 determinant with the last column all ones. local_triangulation.mojo carries a second,
# independently written copy that returns the negation of this test, which is the sign convention its
# caller needs; the two are kept separate rather than sharing one, because the predicate's meaning is
# only well defined together with its caller's convention.


# elementary_geometry.ipp: triangleArea
def triangle_area(l_ab: Float64, l_bc: Float64, l_ca: Float64) -> Float64:
    var s = (l_ab + l_bc + l_ca) / 2.0
    var arg = max(0.0, s * (s - l_ab) * (s - l_bc) * (s - l_ca))
    return sqrt(arg)


# elementary_geometry.ipp: layoutTriangleVertex. Places C on the side which gives CCW winding of ABC.
def layout_triangle_vertex(p_a: Vector2, p_b: Vector2, l_bc: Float64, l_ca: Float64) -> Vector2:
    var l_ab = (p_b - p_a).norm()
    var t_area = triangle_area(l_ab, l_bc, l_ca)
    var h = 2.0 * t_area / l_ab
    var w = (l_ab * l_ab - l_bc * l_bc + l_ca * l_ca) / (2.0 * l_ab)
    var v_abn = (p_b - p_a) / l_ab
    var v_abn_perp = Vector2(-v_abn.y, v_abn.x)
    return p_a + v_abn * w + v_abn_perp * h


# SurfacePoint::faceCoords, a 3-vector of barycentric coordinates
struct Bary3:
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64 = 0.0, y: Float64 = 0.0, z: Float64 = 0.0):
        self.x = x^
        self.y = y^
        self.z = z^

    def __sub__(self, b: Self) -> Self:
        return Bary3(self.x - b.x, self.y - b.y, self.z - b.z)

    def __getitem__(self, i: Int) -> Float64:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        return self.z
