"""This script generates the initial_poses.json of the dodecahedron, fed into the optimization algorithm
It must be adjusted to correspond to the ArUco corners of your object marker"""

# plot a dodecahedron to find out ideal positions
import matplotlib.pyplot as plt
import numpy as np
import cv2

np.set_printoptions(precision=3, suppress=True)


def polygon_coords(n, r=14.933): # 14.933 is the radius of the dodecahedron (from CAD model)
    """Return the coordinates of a regular polygon with n sides and radius r"""
    angles = np.linspace(0, 2*np.pi, n+1)[:-1]
    coords = np.array([r*np.cos(angles), r*np.sin(angles), np.zeros(n)]).T
    # # add in first point to close the polygon
    # coords = np.vstack((coords, coords[0]))
    return coords

def get_edge(coords: np.ndarray, i: int) -> np.ndarray:
    """Return the edge vector between the ith and (i+1)th point"""
    assert i < coords.shape[0]
    # return position and axis
    start_idx = i
    end_idx = (i+1) % coords.shape[0]
    return coords[start_idx], coords[end_idx] - coords[start_idx]


def plot_edge(ax, axis_vector, translation, color='r'):
    """Plot an edge vector"""
    if axis_vector.shape[0] == 2:
        ax.plot([translation[0], translation[0]+axis_vector[0]],
            [translation[1], translation[1]+axis_vector[1]], color=color, linewidth=3)
    else:
        ax.plot([translation[0], translation[0]+axis_vector[0]],
                [translation[1], translation[1]+axis_vector[1]],
                [translation[2], translation[2]+axis_vector[2]], 
                color=color, linewidth=3)


def change_of_basis_transform(x_dash, y_dash, z_dash, vec, x_orig=[1, 0, 0], y_orig=[0, 1, 0], z_orig=[0, 0, 1]):
    """Transform a vector from one basis to another"""
    # create a matrix with the old basis as columns
    old_basis = np.array([x_orig, y_orig, z_orig]).T
    # create a matrix with the new basis as columns
    new_basis = np.array([x_dash, y_dash, z_dash]).T
    # transform the vector
    return new_basis @ np.linalg.inv(old_basis) @ vec



def create_square(translation, dir, plane_vec, length, offset=np.array([0, 0, 0])):
    dir = dir / np.linalg.norm(dir)
    plane_vec = plane_vec / np.linalg.norm(plane_vec)
    normal = np.cross(dir, plane_vec)
    normal = normal / np.linalg.norm(normal)

    vertex0 = translation
    vertex1 = vertex0 + dir * length


    R_90 = cv2.Rodrigues(-np.pi/2 * normal)[0]


    vertex2 = vertex1 + R_90 @ dir * length
    vertex3 = vertex0 + R_90 @ dir * length

    x_dash = dir
    y_dash = R_90 @ dir / np.linalg.norm(R_90 @ dir)
    z_dash = normal

    offset_transformed = change_of_basis_transform(x_dash, y_dash, z_dash, offset)
    
    return np.array([vertex0 + offset_transformed, vertex1 + offset_transformed, vertex2 + offset_transformed, vertex3 + offset_transformed])

def get_rotmat_between_vecs(v1, v2):
    """Get the rotation matrix between two vectors"""
    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate the cross product between the vectors
    v = np.cross(v1, v2)

    # Calculate the dot product between the vectors
    c = np.dot(v1, v2)

    # Construct the skew-symmetric cross product matrix
    skew_cross_product_matrix = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Construct the rotation matrix using Rodrigues' rotation formula
    R = np.identity(3) + skew_cross_product_matrix + np.dot(skew_cross_product_matrix, skew_cross_product_matrix) * (1 - c) / (np.linalg.norm(v) ** 2)

    return R

def rotate_points_around_axis(points, axis_vector, translation, angle_degrees):
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Normalize the axis vector
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Create a rotation matrix
    rotation_matrix = cv2.Rodrigues(angle_radians * axis_vector)[0]

    # Apply translation
    translated_points = points - translation

    # Rotate the points
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    # Apply the translation back
    rotated_points += translation

    return rotated_points


def plot_face(ax, face, color=None):
    plot_face = np.vstack((face, face[0]))
    if face.shape[1] == 2:
        if color is None:
            ax.plot(plot_face[:,0], plot_face[:,1])
        else:
            ax.plot(plot_face[:,0], plot_face[:,1], color=color)
    else:
        if color is None:
            ax.plot(plot_face[:,0], plot_face[:,1], plot_face[:,2])
        else:
            ax.plot(plot_face[:,0], plot_face[:,1], plot_face[:,2], color=color)
    # return color
    return ax.lines[-1].get_color()

# offset = np.array([-0.223, 1.980, 0])
# l = 18
offset = np.array([1, 4, 0]) # aruco marker position in the polygon
l = 12.2 # approximate aruco marker size
r_polygon = 12.8

# plot a polygon with 5 sides
fig = plt.figure(figsize=(10, 10))
f0 = polygon_coords(5, r=r_polygon)
dihedral_angle = 2*np.arctan((1+np.sqrt(5))/2)
print(f'Dihedral angle [deg]: {np.rad2deg(dihedral_angle)}')

ax = fig.add_subplot(111, projection='3d')
t, r = get_edge(f0, 0)



square0 = create_square(t, r, get_edge(f0, 4)[1], l, offset=offset)

# plot all 12 faces
idx_range = [0, 0, 1, 4, 1, 4, 3, 0, 2, 0, 2, 1]
# id_range = [0, 89, 90, 91, 94, 92, 96, 95, 93, 98, 97, 99]
# id_range = [10, 0, 1, 2, 5, 3, 7, 6, 4, 9, 8, -1]
# now range from 11 to 11+12
id_start = 54
n_id = 11
ids = [id_start + i for i in range(n_id)]
faces_idx = [10, 0, 1, 2, 5, 3, 7, 6, 4, 9, 8, 11]

id_range = [ids[i] if len(ids) > i else -1 for i in faces_idx] 
sq_roll = [0, 0, -2, 2, -2, 2, -2, 0, 2, 0, 2, 0]
sq_inverse = [False, True, True, True, True, True, False, True, False, True, False, True]
angles = [0, np.rad2deg(dihedral_angle), -np.rad2deg(dihedral_angle), np.rad2deg(dihedral_angle), -np.rad2deg(dihedral_angle), np.rad2deg(dihedral_angle), -np.rad2deg(dihedral_angle), np.rad2deg(dihedral_angle), -np.rad2deg(dihedral_angle), np.rad2deg(dihedral_angle), -np.rad2deg(dihedral_angle), np.rad2deg(dihedral_angle)]
# rotate_points_through

faces = []
squares = []
n_faces = 12
for i in range(n_faces):
    f = faces[i-1] if i > 0 else f0
    t, r = get_edge(f, idx_range[i])
    new_face = rotate_points_around_axis(f, r, t, angles[i])
    t, r = get_edge(new_face, idx_range[i])
    # c = plot_face(ax, new_face)
    if i == 0:
        new_square = square0
    elif 0 < i < 6:
        s = squares[i-1]
        new_square = rotate_points_around_axis(s, r, t, angles[i])
    else:
        t1, r1 = get_edge(new_face, 1)
        r2 = get_edge(new_face, 0)[1]
        new_square = create_square(t1, r1, r2, l, offset=offset)
    # plot_face(ax, new_square, color=c)
    # roll square indices and reverse if necessary
    if sq_inverse[i]:
        new_square = np.flip(new_square, axis=0)
    new_square = np.roll(new_square, sq_roll[i], axis=0)

    # if any new_face is equal to a previous face, raise an error
    for face in faces:
        # sort points by x, y, z
        new_face_sorted = np.sort(new_face, axis=0)
        face_sorted = np.sort(face, axis=0)
        if np.allclose(new_face_sorted, face_sorted):
            raise ValueError('face already exists')
    faces.append(new_face)
    squares.append(new_square)


# get the rotation matrix such that v is the new x-axis
R = get_rotmat_between_vecs(squares[faces_idx.index(0)][2] - squares[faces_idx.index(0)][3], [0, 1, 0])

# rotate everything by R
for i in range(len(faces)):
    faces[i] = np.dot(faces[i], R.T)
    squares[i] = np.dot(squares[i], R.T)


# plot all
for id, face, square in zip(id_range, faces, squares):
    c = plot_face(ax, square)
    ax.text(square[:,0].mean(), square[:,1].mean(), square[:,2].mean(), str(id))
    plot_face(ax, face, color=c)
    # write corner index on each corner of square
    for j in range(square.shape[0]):
        ax.text(square[j,0], square[j,1], square[j,2], str(j))

# plot the IMU location (which is now at the origin)
ax.scatter(0, 0, 0, color='r', s=100)


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.axis('equal')
# # look from front
ax.view_init(30, 45)

# save to out and close
plt.tight_layout()
if False:
    plt.savefig('out.png', dpi=300)
    plt.close()
else:
    plt.show()

# dump into json with {id: square_corners, ...}
import json
json_dict = {}
for i in range(len(squares)):
    # skip id -1
    if id_range[i] == -1:
        continue
    # divide squares by 1000 to get m
    squares_m = squares[i] / 1000
    json_dict[id_range[i]] = squares_m.tolist()
with open('initial_poses.json', 'w') as f:
    json.dump(json_dict, f, indent=4)