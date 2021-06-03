import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import cv2

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 255, 0)
yellow = (0, 255, 255)
magenta = (255, 0, 255)
white = (255, 255, 255)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def draw_hand_skel_front(points, mask, ratio):

    points = points.copy()
    
    mask_w, mask_h = mask.shape[1], mask.shape[0]
    hand_x = points[:, 0] * ratio
    hand_y = points[:, 1]
    hand_x -= np.min(hand_x)
    hand_y -= np.min(hand_y)

    x1 = np.min(hand_x)
    y1 = np.min(hand_y)
    x2 = np.max(hand_x)
    y2 = np.max(hand_y)

    if x2 - x1 >= y2 - y1:
        s = mask_w/(x2 - x1)
    else:
        s = mask_h/(y2 - y1)

    hand_x *= s
    hand_y *= s

    hand_x *= 0.9 
    hand_x += (mask_w - np.max(hand_x))/2
    x = hand_x.astype(int)

    hand_y *= 0.9 
    hand_y += (mask_h - np.max(hand_y))/2
    y = hand_y.astype(int)

    c = (255, 255, 255)
    t = int(np.sqrt(max(mask.shape[0], mask.shape[1])))

    k1 = 0.4
    k2 = (1-k1)*255/3
    alpha = 0.2

    n = 0
    color = [red, yellow, green, blue, magenta]
    for i in range(0, 5):

        overlay = mask.copy()

        c = tuple(int(j * k1) for j in color[i])
        cv2.line(mask, (x[0], y[0]), (x[n*4+1], y[n*4+1]), c, t)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        c = tuple(int(j + k2) if j != 0 else 0 for j in c)
        cv2.line(mask, (x[n*4+1], y[n*4+1]), (x[n*4+2], y[n*4+2]), c, t)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        c = tuple(int(j + k2) if j != 0 else 0 for j in c)
        cv2.line(mask, (x[n*4+2], y[n*4+2]), (x[n*4+3], y[n*4+3]), c, t)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        c = tuple(int(j + k2) if j != 0 else 0 for j in c)
        cv2.line(mask, (x[n*4+3], y[n*4+3]), (x[n*4+4], y[n*4+4]), c, t)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        n += 1

    return mask


def draw_hand_skel_side(points, mask, ratio):

    points = points.copy()
    
    mask_w, mask_h = mask.shape[1], mask.shape[0]
    hand_z = points[:, 2] * ratio
    hand_y = points[:, 1]
    hand_z -= np.min(hand_z)
    hand_y -= np.min(hand_y)

    z1 = np.min(hand_z)
    y1 = np.min(hand_y)
    z2 = np.max(hand_z)
    y2 = np.max(hand_y)

    if z2 - z1 >= y2 - y1:
        s = mask_w/(z2 - z1)
    else:
        s = mask_h/(y2 - y1)

    hand_z *= s
    hand_y *= s

    hand_z *= 0.9 
    hand_z += (mask_w - np.max(hand_z))/2
    z = hand_z.astype(int)

    hand_y *= 0.9 
    hand_y += (mask_h - np.max(hand_y))/2
    y = hand_y.astype(int)

    c = (255, 255, 255)
    t = int(np.sqrt(max(mask.shape[0], mask.shape[1])))

    k1 = 0.4
    k2 = (1-k1)*255/3
    alpha = 0.2

    n = 0
    color = [red, yellow, green, blue, magenta]
    for i in range(0, 5):

        overlay = mask.copy()

        c = tuple(int(j * k1) for j in color[i])
        cv2.line(mask, (z[0], y[0]), (z[n*4+1], y[n*4+1]), c, t, lineType=cv2.LINE_AA)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        c = tuple(int(j + k2) if j != 0 else 0 for j in c)
        cv2.line(mask, (z[n*4+1], y[n*4+1]), (z[n*4+2], y[n*4+2]), c, t, lineType=cv2.LINE_AA)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        c = tuple(int(j + k2) if j != 0 else 0 for j in c)
        cv2.line(mask, (z[n*4+2], y[n*4+2]), (z[n*4+3], y[n*4+3]), c, t, lineType=cv2.LINE_AA)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        c = tuple(int(j + k2) if j != 0 else 0 for j in c)
        cv2.line(mask, (z[n*4+3], y[n*4+3]), (z[n*4+4], y[n*4+4]), c, t, lineType=cv2.LINE_AA)
        mask = cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0)

        n += 1

    return mask


def rotate_3d(points):
    a = points[9] - points[0]
    b = np.array([0, -1, 0])
    rot_matxy = rotation_matrix_from_vectors(a, b)
    points = np.dot(points, rot_matxy.T)

    a = points[13] - points[9]
    theta = np.arctan2(a[2], a[0])
    c = np.cos(theta)
    s = np.sin(theta)
    rot_matzy = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    points = np.dot(points, rot_matzy.T)

    if estimate_palm_normal(points):
        theta = np.pi
        c = np.cos(theta)
        s = np.sin(theta)
        rot_mat_left_hand = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        points = np.dot(points, rot_mat_left_hand)

    return points


def view_points(points):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    r = 2

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2])  # plot the point (2,3,4) on the figure
    connect_hand_points(ax1, points)

    ax1.set_xlim3d(-r, r)
    ax1.set_ylim3d(-r, r)
    ax1.set_zlim3d(-r, r)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=-90, azim=-90)


def estimate_palm_normal(points):

    f1 = points[8][2] > points[5][2]
    f2 = points[12][2] > points[9][2]
    f3 = points[16][2] > points[13][2]
    f4 = points[20][2] > points[17][2]
    f5 = points[4][2] > points[2][2]

    condts = [f1, f2, f3, f4, f5]
    count = condts.count(True)

    if count >= 4:
        return True
    else:
        return False


def process_3d(points):

    fig = plt.figure()

    ax1 = plt.subplot(3, 4, 1, projection='3d')
    ax4 = plt.subplot(3, 4, 5, projection='3d')
    ax7 = plt.subplot(3, 4, 9, projection='3d')

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax1, points)
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax4, points)
    ax7.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax7, points)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")

    ax7.set_xlabel("x")
    ax7.set_ylabel("y")
    ax7.set_zlabel("z")

    ax1.view_init(elev=-90, azim=-90)
    ax4.view_init(elev=0, azim=-90)
    ax7.view_init(elev=0, azim=0)

    a = points[9] - points[0]
    b = np.array([0, -1, 0])
    rot_matxy = rotation_matrix_from_vectors(a, b)
    points = np.dot(points, rot_matxy.T)

    ax2 = plt.subplot(3, 4, 2, projection='3d')
    ax5 = plt.subplot(3, 4, 6, projection='3d')
    ax8 = plt.subplot(3, 4, 10, projection='3d')

    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax2, points)
    ax5.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax5, points)
    ax8.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax8, points)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_zlabel("z")

    ax8.set_xlabel("x")
    ax8.set_ylabel("y")
    ax8.set_zlabel("z")

    ax2.view_init(elev=-90, azim=-90)
    ax5.view_init(elev=0, azim=-90)
    ax8.view_init(elev=0, azim=0)

    a = points[13] - points[9]
    theta = np.arctan2(a[2], a[0])
    c = np.cos(theta)
    s = np.sin(theta)
    rot_matzy = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    points = np.dot(points, rot_matzy.T)

    ax3 = plt.subplot(3, 4, 3, projection='3d')
    ax6 = plt.subplot(3, 4, 7, projection='3d')
    ax9 = plt.subplot(3, 4, 11, projection='3d')

    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax3, points)
    ax6.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax6, points)
    ax9.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax9, points)

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_zlabel("z")

    ax9.set_xlabel("x")
    ax9.set_ylabel("y")
    ax9.set_zlabel("z")

    ax3.view_init(elev=-90, azim=-90)
    ax6.view_init(elev=0, azim=-90)
    ax9.view_init(elev=0, azim=0)

    if estimate_palm_normal(points):
        theta = np.pi
        c = np.cos(theta)
        s = np.sin(theta)
        rot_matzy = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        points = np.dot(points, rot_matzy.T)

    ax10 = plt.subplot(1, 4, 4, projection='3d')
    ax10.scatter(points[:, 0], points[:, 1], points[:, 2], c="black")
    connect_hand_points(ax10, points)
    ax10.set_xlabel("x")
    ax10.set_ylabel("y")
    ax10.set_zlabel("z")
    ax10.view_init(elev=-90, azim=-90)

    plt.tight_layout()
    plt.show()


def connect_hand_points(pltn, points):
    color = ['red', 'orange', 'green', 'darkblue', 'purple']
    for i in range(0, 5):
        c = "black"
        connect_points(pltn, points[0], points[i * 4 + 1], c)
        if i != 0:
            c = color[i]
        connect_points(pltn, points[i * 4 + 1], points[i * 4 + 2], c)
        if i == 0:
            c = color[i]
        connect_points(pltn, points[i * 4 + 2], points[i * 4 + 3], c)
        connect_points(pltn, points[i * 4 + 3], points[i * 4 + 4], c)

    connect_points(pltn, points[2], points[5], 'black')
    connect_points(pltn, points[5], points[9], 'black')
    connect_points(pltn, points[9], points[13], 'black')
    connect_points(pltn, points[13], points[17], 'black')


def connect_points(pltn, ptn1, ptn2, c):
    x1, x2 = ptn1[0], ptn2[0]
    y1, y2 = ptn1[1], ptn2[1]
    z1, z2 = ptn1[2], ptn2[2]
    pltn.plot([x1, x2], [y1, y2], [z1, z2], 'k-', color=c)


def subtract_offset(points):

    points[:, 0] -= points[0][0]
    points[:, 1] -= points[0][1]
    points[:, 2] -= points[0][2]

    return points


def normalise(points):
    a = points[9] - points[0]

    points /= np.linalg.norm(a)

    return points


def test():

    file = "Basic words - Auslan"
    # file = "handshapes"
    # file = "WIN_20210418_10_11_51_Pro"
    # file = "handshapes"

    df = pd.read_csv(file + "/right_hand_keypoints.csv")
    df = np.array(df)[:, 1:]

    for i in range(50):
        rand_idx = random.randint(0, len(df) - 1)
        points = df[rand_idx]

        x = points[0::3] - points[0]
        y = points[1::3] - points[1]
        z = points[2::3] - points[2]

        points = np.ones((21, 3))
        points[:, 0] = x * 1280/720
        points[:, 1] = y
        points[:, 2] = z * 1280/720

        # points = subtract_offset(points)
        # points = rotate_3d(points)
        # points = normalise(points)

        process_3d(points)
        plt.show()


def main():
    test()


if __name__ == "__main__":
    main()


