import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __str__(self):
        return f"Circle(center={self.center}, radius={self.radius})"

    def relation_with_circle(self, other_circle):
        distance = math.sqrt((self.center.x - other_circle.center.x) ** 2 + (self.center.y - other_circle.center.y) ** 2)
        radius_sum = self.radius + other_circle.radius
        radius_diff = abs(self.radius - other_circle.radius)

        if distance == 0:
            if self.radius == other_circle.radius:
                return "重合"
            else:
                return "包含"
        elif distance == radius_sum:
            return "相切"
        elif distance < radius_sum:
            if distance > radius_diff:
                return "相交"
            else:
                return "包含"
        else:
            return "相离"

# 任务2-4
points = [Point(i, i) for i in range(1, 4)]
print("Original points:", ", ".join(str(point) for point in points))

# 任务4
points[0], points[2] = points[2], points[0]
print("Swapped points:", ", ".join(str(point) for point in points))

# 任务5-6
circles = [Circle(Point(i, i), 1 + 2 * i) for i in range(3)]
for circle in circles:
    print(circle)

# 测试第一个圆和第二个圆的关系
print("Relation:", circles[0].relation_with_circle(circles[1]))
