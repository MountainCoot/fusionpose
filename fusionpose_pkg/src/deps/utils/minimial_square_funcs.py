import numpy as np
import cv2
from scipy.optimize import least_squares

def find_orthogonal_vector(v, angle=0):
    if np.allclose(v, [0, 0, 0]):
        raise ValueError("Input vector cannot be the zero vector.")
    
    a = np.array([1, 0, 0]) if not np.allclose(v[:2], [0, 0]) else np.array([0, 1, 0])
    u = np.cross(v, a)

    if angle != 0:
        R = cv2.Rodrigues(v * angle)[0]
        u = R @ u

    if np.linalg.norm(u) > 1e-6:
        return u / np.linalg.norm(u)
    else:
        raise ValueError("Degenerate case encountered.")

def opt_r14_star(p1, p2, tolerance=1e-6):
    r12 = p2 - p1

    def opt_fun(angle):
        r14_star = find_orthogonal_vector(r12, angle)
        return r14_star[2]

    res = least_squares(opt_fun, 0, bounds=(0, 2 * np.pi))
    angle = res.x[0]
    r14_star = find_orthogonal_vector(r12, angle)
    
    p4_star = p1 + r14_star

    assert np.abs(np.linalg.norm(p4_star - p1) - 1.0) < tolerance, f"Distance |p4_star - p1| must be 1."
    assert np.abs(np.dot(p4_star - p1, r12)) < tolerance, f"Angle between p4_star - p1 and r12 must be 90 degrees."

    return r14_star

def points_to_description(p1, p2, p4, tolerance=1e-6):
    r12 = p2 - p1
    r14 = p4 - p1

    assert np.abs(np.linalg.norm(r12) - 1.0) < tolerance, "Distance |p2 - p1| must be 1."
    assert np.abs(np.linalg.norm(r14) - 1.0) < tolerance, "Distance |p4 - p1| must be 1."
    assert np.abs(np.dot(r12, r14)) < tolerance, "Angle between r12 and r14 must be 90 degrees."

    r14_star = opt_r14_star(p1, p2, tolerance=tolerance)
    r = r12 / np.linalg.norm(r12)

    def residual(angle):
        R = cv2.Rodrigues(r * angle)[0]
        r14_star_rotated = R @ r14_star
        return r14_star_rotated - r14
    
    res = least_squares(residual, 0, bounds=(-np.pi, np.pi))
    angle = res.x[0]
    if angle < 0:
        angle += 2 * np.pi

    r = r12 / np.linalg.norm(r12) * angle
    R = cv2.Rodrigues(r)[0]
    p4_star_rotated = R @ r14_star + p1

    assert np.allclose(p4_star_rotated, p4, atol=tolerance), f"p4_star_rotated: {p4_star_rotated}, p4: {p4}"

    if np.abs(angle) < 1e-6:
        angle += 1e-6
        r = r12 / np.linalg.norm(r12) * angle

    return p1, r

def description_to_points(p1, r):
    angle = np.linalg.norm(r)
    r12 = r / angle
    p2 = p1 + r12

    r14_star = opt_r14_star(p1, p2)

    R = cv2.Rodrigues(r)[0]
    p4 = R @ r14_star + p1

    p3 = p4 + p2 - p1

    return p1, p2, p3, p4
