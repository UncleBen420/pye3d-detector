import numpy as np
from numpy import cbrt
from math import sqrt, cos, sin, acos
from ..geometry.primitives import Conic, Conicoid, Circle


def sign(val):
    """Type-safe signum function, returns -1, 0, or 1."""
    return (0 < val) - (val < 0)


def solve_0(a):
    """Solve a = 0."""
    if a == 0:
        return 0
    else:
        raise RuntimeError("No solution")


def solve_1(a, b):
    """Solve ax + b = 0."""
    if a == 0:
        return solve_0(b)
    return -b / a


def solve_2(a, b, c):
    """Solve ax^2 + bx + c = 0."""
    if a == 0:
        root = solve_1(b, c)
        return (root, root)

    det = b * b - 4 * a * c
    if det < 0:
        raise RuntimeError("No solution")

    q = -0.5 * (b + (1 if b >= 0 else -1) * sqrt(det))
    return (q / a, c / q)


def solve_3(a, b, c, d):
    """Solve ax^3 + bx^2 + cx + d = 0."""
    if a == 0:
        roots = solve_2(b, c, d)
        return (roots[0], roots[1], roots[1])

    p = b / a
    q = c / a
    r = d / a

    u = q - (p * p) / 3
    v = r - p * q / 3 + 2 * p * p * p / 27
    j = 4 * u * u * u / 27 + v * v

    M = float('inf')
    sqrtM = sqrt(M)
    cbrtM = cbrt(M)

    if b == 0 and c == 0:
        return (cbrt(-d), cbrt(-d), cbrt(-d))
    if abs(p) > 27 * cbrtM:
        return (-p, -p, -p)
    if abs(q) > sqrtM:
        return (-cbrt(v), -cbrt(v), -cbrt(v))
    if abs(u) > 3 * cbrtM / 4:
        return (cbrt(4) * u / 3, cbrt(4) * u / 3, cbrt(4) * u / 3)

    if j > 0:
        w = sqrt(j)
        if v > 0:
            y = (u / 3) * cbrt(2 / (w + v)) - cbrt((w + v) / 2) - p / 3
        else:
            y = cbrt((w - v) / 2) - (u / 3) * cbrt(2 / (w - v)) - p / 3
        return (y, y, y)
    else:
        s = sqrt(-u / 3)
        t = -v / (2 * s * s * s)
        k = acos(t) / 3
        y1 = 2 * s * cos(k) - p / 3
        y2 = s * (-cos(k) + sqrt(3) * sin(k)) - p / 3
        y3 = s * (-cos(k) - sqrt(3) * sin(k)) - p / 3
        return (y1, y2, y3)


def unproject_conicoid(a, b, c, f, g, h, u, v, w, focal_length, circle_radius):
    """Unproject conicoid to find two 3D circles."""
    # Solve for eigenvalues (lambdas) using the cubic solver
    lambda_ = np.array(solve_3(1.0, -(a + b + c),
                               b * c + c * a + a * b - f * f - g * g - h * h,
                               -(a * b * c + 2 * f * g * h - a * f * f - b * g * g - c * h * h)))

    # Ensure lambda ordering
    assert lambda_[0] >= lambda_[1]
    assert lambda_[1] > 0
    assert lambda_[2] < 0
    
    # Calculate plane parameters l, m, n
    n = sqrt((lambda_[1] - lambda_[2]) / (lambda_[0] - lambda_[2]))
    m = 0.0
    l = sqrt((lambda_[0] - lambda_[1]) / (lambda_[0] - lambda_[2]))
    
    # Compute T1 rotation matrix
    T1 = np.zeros((3, 3))
    t1 = (b - lambda_) * g - f * h
    t2 = (a - lambda_) * f - g * h
    t3 = -(a - lambda_) * (t1 / t2) / g - h / g
    mi = 1 / np.sqrt(1 + (t1 / t2) ** 2 + t3 ** 2)
    T1[0] = (t1 / t2) * mi  # li
    T1[1] = mi               # mi
    T1[2] = t3 * mi          # ni
    
    # Ensure right-handed coordinate system
    if np.dot(np.cross(T1[0], T1[1]), T1[2]) < 0:
        T1 = -T1
    
    # Compute T2 translation
    T2_translation = -(u * T1[0] + v * T1[1] + w * T1[2]) / lambda_
    
    solutions = []
    ls = [l, -l]
    
    for l in ls:
        # Circle normal in image space (gaze vector)
        gaze = T1 @ np.array([l, m, n])
        
        # Compute T3 rotation matrix
        T3 = np.zeros((3, 3))
        if l == 0:
            assert abs(n - 1) < 1e-10
            T3 = np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
        else:
            T3 = np.array([[0, -n * sign(l), l],
                          [sign(l), 0, 0],
                          [0, abs(l), n]])
        
        # Calculate circle center
        A = lambda_ @ (T3[:, 0] ** 2)
        B = lambda_ @ (T3[:, 0] * T3[:, 2])
        C = lambda_ @ (T3[:, 1] * T3[:, 2])
        D = lambda_ @ (T3[:, 2] ** 2)
        
        center_in_Xprime = np.zeros(3)
        center_in_Xprime[2] = A * circle_radius / sqrt(B * B + C * C - A * D)
        center_in_Xprime[0] = -B / A * center_in_Xprime[2]
        center_in_Xprime[1] = -C / A * center_in_Xprime[2]
        
        # Apply transformations
        T0_translation = np.array([0, 0, focal_length])
        center = T0_translation + T1 @ (T2_translation + T3 @ center_in_Xprime)
        
        # If center is behind camera, try the other solution
        if center[2] < 0:
            center_in_Xprime = -center_in_Xprime
            center = T0_translation + T1 @ (T2_translation + T3 @ center_in_Xprime)
        
        # Normalize gaze and ensure it points toward the camera
        if np.dot(gaze, center) > 0:
            gaze = -gaze
        gaze = gaze / np.linalg.norm(gaze)
        
        solutions.append({
            'center': center,
            'normal': gaze,
            'radius': circle_radius
        })

    return solutions[0], solutions[1]


def unproject_ellipse(ellipse, focal_length, radius=1.0):

    try:

        conic = Conic(ellipse)
        pupil_cone = Conicoid(conic, [0, 0, -focal_length])

        circle_A, circle_B = unproject_conicoid(
            pupil_cone.A,
            pupil_cone.B,
            pupil_cone.C,
            pupil_cone.F,
            pupil_cone.G,
            pupil_cone.H,
            pupil_cone.U,
            pupil_cone.V,
            pupil_cone.W,
            focal_length,
            radius
        )

        # Check for invalid values
        values = np.concatenate([
            [circle_A['radius']], circle_A['center'], circle_A['normal'],
            [circle_B['radius']], circle_B['center'], circle_B['normal']
        ])
        if np.isnan(values).any():
            return False

        circle_A = Circle(
                    center=(circle_A['center'][0], circle_A['center'][1], circle_A['center'][2]),
                    normal=(circle_A['normal'][0], circle_A['normal'][1], circle_A['normal'][2]),
                    radius=circle_A['radius']
                    )
        circle_B = Circle(
                    center=(circle_B['center'][0], circle_B['center'][1], circle_B['center'][2]),
                    normal=(circle_B['normal'][0], circle_B['normal'][1], circle_B['normal'][2]),
                    radius=circle_B['radius']
                    )

        return [circle_A, circle_B]

    except AssertionError:
        return False
