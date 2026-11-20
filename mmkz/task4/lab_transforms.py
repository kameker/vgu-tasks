from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


plt.switch_backend("Agg")
FOLDER = Path(__file__).resolve().parent
RESULTS = FOLDER / "results"
RESULTS.mkdir(exist_ok=True)


def load(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть {path}")
    return image


def save(name, images, columns):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, image) in zip(axes.flat, images):
        if image.ndim == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
    fig.savefig(RESULTS / name, dpi=130)
    plt.close(fig)


def exercise_1():
    image = load(FOLDER / "face.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    inverted = 255 - gray
    laplace = -cv2.Laplacian(inverted, cv2.CV_32F, ksize=9)
    laplace = cv2.GaussianBlur(laplace, (9, 9), 2)
    result = image.copy()
    for start, end in ((0.28, 0.43), (0.58, 0.73)):
        left = int(w * start)
        right = int(w * end)
        top = int(h * 0.43)
        bottom = int(h * 0.49)
        region = laplace[top:bottom, left:right]
        _, _, _, point = cv2.minMaxLoc(region)
        cv2.circle(result, (left + point[0], top + point[1]), max(4, w // 80), (0, 0, 255), 2)
    response = cv2.normalize(laplace, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save("exercise_1.png", [("Исходное фото", image), ("Отклик оператора Лапласа", response), ("Найденные зрачки", result)], 3)


def polar(image, center):
    return cv2.warpPolar(image, (360, 360), center, 360, cv2.WARP_POLAR_LOG + cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS)


def exercise_2():
    square = np.zeros((400, 400), np.uint8)
    cv2.rectangle(square, (120, 120), (280, 280), 255, 3)
    circle = np.zeros_like(square)
    cv2.circle(circle, (200, 200), 110, 255, 3)
    images = []
    for name, source, center in (("a. Вершина квадрата", square, (120, 120)),
                                 ("b. Внутри окружности", circle, (290, 200)),
                                 ("c. Снаружи окружности", circle, (330, 200))):
        marked = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        cv2.circle(marked, center, 5, (0, 0, 255), -1)
        images.extend([(name, marked), ("Лог-полярное преобразование", polar(source, center))])
    save("exercise_2.png", images, 2)


def exercise_4():
    images = []
    transformed = []
    for size, angle in ((180, 0), (100, 0), (180, 25), (100, 25)):
        source = np.zeros((400, 400), np.uint8)
        points = cv2.boxPoints(((200, 200), (size, size), angle)).astype(np.int32)
        cv2.fillConvexPoly(source, points, 255)
        mapped = polar(source, (200, 200))
        images.append((f"Сторона {size}, угол {angle}°", source))
        transformed.append(mapped)
    reference = transformed[0].astype(np.float32)
    reference_edge = np.flatnonzero(transformed[0][180])[-1]
    center_shift = 180 - int(reference_edge)
    for index, mapped in enumerate(transformed):
        images.append((f"Лог-полярное {index + 1}", mapped))
    for index, mapped in enumerate(transformed):
        shift, _ = cv2.phaseCorrelate(reference, mapped.astype(np.float32))
        dx, dy = np.rint(shift).astype(int)
        aligned = np.roll(mapped, -dy, axis=0)
        matrix = np.float32([[1, 0, center_shift - dx], [0, 1, 0]])
        aligned = cv2.warpAffine(aligned, matrix, (360, 360), borderValue=255)
        images.append((f"Совмещение {index + 1}", aligned))
    save("exercise_4.png", images, 4)


def exercise_7():
    images = []
    for name in ("i1.jpg", "i2.jpg", "i3.jpg", "i4.jpg"):
        gray = cv2.cvtColor(load(FOLDER.parent / "task2" / name), cv2.COLOR_BGR2GRAY)
        images.extend([(name, gray), (f"{name}, equalizeHist", cv2.equalizeHist(gray))])
    save("exercise_7.png", images, 2)


exercise_1()
exercise_2()
exercise_4()
exercise_7()
