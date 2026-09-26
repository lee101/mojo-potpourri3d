"""geometry-central's Vector2 / Vector3, by value.

Only the operations the heat-method kernels actually use. Kept as plain structs
so the arithmetic reads like the C++ it is a port of.
"""

from std.math import sqrt, cos, sin, fabs, atan2

struct Vec2:
    var x: Float64
    var y: Float64

    def __init__(out self, x: Float64, y: Float64 = 0.0):
        self.x = x
        self.y = y

    def __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y)

    def __mul__(self, s: Float64) -> Self:
        return Self(self.x * s, self.y * s)

    def __truediv__(self, s: Float64) -> Self:
        return Self(self.x / s, self.y / s)

    def __neg__(self) -> Self:
        return Self(-self.x, -self.y)

    def dot(self, other: Self) -> Float64:
        return self.x * other.x + self.y * other.y

    def cross(self, other: Self) -> Float64:
        return self.x * other.y - self.y * other.x

    def norm2(self) -> Float64:
        return self.x * self.x + self.y * self.y

    def norm(self) -> Float64:
        return sqrt(self.x * self.x + self.y * self.y)

    # Vector2::rotate90
    def rotate90(self) -> Self:
        return Self(-self.y, self.x)

    # Vector2::normalize
    def normalize(self) -> Self:
        return self * (1.0 / sqrt(self.x * self.x + self.y * self.y))

    # Vector2::normalizeCutoff, mag = 0
    def normalize_cutoff(self) -> Self:
        var length = sqrt(self.x * self.x + self.y * self.y)
        if length <= 0.0:
            length = 1.0
        return self * (1.0 / length)

    # Vector2::fromAngle
    @staticmethod
    def from_angle(theta: Float64) -> Self:
        return Self(cos(theta), sin(theta))

    def arg(self) -> Float64:
        return atan2(self.y, self.x)

    def inv(self) -> Self:
        var d = self.norm2()
        return Self(self.x / d, -self.y / d)

    # Complex multiplication, used where geometry-central writes Vector2 as a
    # complex (Vector2 * Vector2 is the scalar product there).
    def cmul(self, other: Self) -> Self:
        return Self(
            self.x * other.x - self.y * other.y,
            self.x * other.y + self.y * other.x,
        )

    # Complex division, used where geometry-central writes Vector2 as a complex.
    def cdiv(self, other: Self) -> Self:
        var d = other.norm2()
        return Self(
            (self.x * other.x + self.y * other.y) / d,
            (self.y * other.x - self.x * other.y) / d,
        )


struct Vec3:
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64 = 0.0, z: Float64 = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: Float64) -> Self:
        return Self(self.x * s, self.y * s, self.z * s)

    def dot(self, other: Self) -> Float64:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Self) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> Float64:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> Self:
        return self * (1.0 / sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def remove_component(self, n: Self) -> Self:
        return self - n * self.dot(n)

    # Vector3::rotateAround
    def rotate_around(self, axis: Self, theta: Float64) -> Self:
        var c = cos(theta)
        var s = sin(theta)
        return self * c + axis.cross(self) * s + axis * (axis.dot(self) * (1.0 - c))

    # Vector3::buildTangentBasis
    def build_tangent_basis(self) -> Tuple[Vec3, Vec3]:
        var unit_dir = self.normalize()
        var test = Vec3(1.0, 0.0, 0.0)
        if fabs(test.dot(unit_dir)) > 0.9:
            test = Vec3(0.0, 1.0, 0.0)
        var basis_x = test.cross(unit_dir).normalize()
        var basis_y = unit_dir.cross(basis_x).normalize()
        return (basis_x, basis_y)
