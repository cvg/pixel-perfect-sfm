import matplotlib.pyplot as plt
import numpy as np
import PIL
import plotly.graph_objects as go


def draw_epipolar_line(line, imshape, axis, color="b"):
    h, w = imshape[:2]
    # Intersect line with lines representing image borders.
    X1 = np.cross(line, [1, 0, -1])
    X1 = X1[:2] / X1[2]
    X2 = np.cross(line, [1, 0, -w])
    X2 = X2[:2] / X2[2]
    X3 = np.cross(line, [0, 1, -1])
    X3 = X3[:2] / X3[2]
    X4 = np.cross(line, [0, 1, -h])
    X4 = X4[:2] / X4[2]

    # Find intersections which are not outside the image,
    # which will therefore be on the image border.
    Xs = [X1, X2, X3, X4]
    Ps = []
    for p in range(4):
        X = Xs[p]
        if (0 <= X[0] <= (w + 1e-6)) and (0 <= X[1] <= (h + 1e-6)):
            Ps.append(X)
            if len(Ps) == 2:
                break

    # Plot line, if it's visible in the image.
    if len(Ps) == 2:
        axis.plot([Ps[0][0], Ps[1][0]], [Ps[0][1], Ps[1][1]],
                  color, linestyle="dashed")


def get_line(F, kp):
    hom_kp = np.array([list(kp) + [1.0]]).transpose()
    return np.dot(F, hom_kp)


def plot_epipolar_lines(pts0, pts1, F, color="b"):
    axes = plt.gcf().axes
    assert(len(axes) == 2)
    is_first = True
    for a, kps in zip(axes, [pts1, pts0]):
        _, w = a.get_xlim()
        h, _ = a.get_ylim()

        imshape = (h+0.5, w+0.5)
        for i in range(kps.shape[0]):
            if is_first:
                line = get_line(F.transpose(), kps[i])[:, 0]
            else:
                line = get_line(F, kps[i])[:, 0]
            draw_epipolar_line(line, imshape, a, color=color)
        is_first = False


def init_image(image_path, height=800):
    image = PIL.Image.open(image_path).convert("RGB")
    fig = go.Figure(go.Image(z=np.array(image)))
    fig.update_traces(
        hovertemplate=None, hoverinfo="skip")
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False, showticklabels=False),
        yaxis=dict(visible=False, showticklabels=False),
        margin=dict(l=0, r=0, b=0, t=0, pad=0))  # noqa E741
    return fig


def plot_points2D(fig, pts2D, color='rgba(0, 0, 255, 0.5)', ps=6):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if isinstance(pts2D, list):
        pts2D = np.array(pts2D)
    x, y = pts2D.T
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name="",
                             marker=dict(color=color)))
