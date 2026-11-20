from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


plt.switch_backend("Agg")

FOLDER = Path(__file__).resolve().parent
RESULTS = FOLDER / "results"


def read_image(path, mode=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), mode)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение {path}")
    return image


def save_grid(filename, images, columns):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.5 * rows),
                             squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.axis("off")
    for axis, (title, image) in zip(axes.flat, images):
        if image.ndim == 3:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
        else:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title)
    fig.savefig(RESULTS / filename, dpi=130)
    plt.close(fig)


def exercise_12(image):
    resized = image
    pyramid = image
    resize_images = []
    pyramid_images = []
    for step in range(1, 4):
        height, width = resized.shape[:2]
        size = ((width + 1) // 2, (height + 1) // 2)
        resized = cv2.resize(resized, size, interpolation=cv2.INTER_LINEAR)
        pyramid = cv2.pyrDown(pyramid)
        resize_images.append((f"resize {step}: {size[0]} × {size[1]}", resized))
        pyramid_images.append((f"pyrDown {step}: {size[0]} × {size[1]}", pyramid))
    save_grid("exercise_12.png", resize_images + pyramid_images, 3)


def exercise_13(gray):
    operations = [
        ("BINARY", cv2.THRESH_BINARY),
        ("BINARY_INV", cv2.THRESH_BINARY_INV),
        ("TRUNC", cv2.THRESH_TRUNC),
        ("TOZERO", cv2.THRESH_TOZERO),
        ("TOZERO_INV", cv2.THRESH_TOZERO_INV),
    ]
    images = [("Исходное изображение", gray)]
    for name, operation in operations:
        _, result = cv2.threshold(gray, 128, 255, operation)
        images.append((f"{name}, T = 128", result))
    save_grid("exercise_13.png", images, 3)

    images = []
    for c in (5, 0, -5):
        for name, method in (("Mean", cv2.ADAPTIVE_THRESH_MEAN_C),
                             ("Gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)):
            for mode, operation in operations[:2]:
                result = cv2.adaptiveThreshold(gray, 255, method, operation, 31, c)
                images.append((f"{name}, {mode}\nC = {c}", result))
    save_grid("exercise_13_adaptive.png", images, 4)


def exercise_14(image, gray):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 110)
    mean_shift = cv2.pyrMeanShiftFiltering(image, 18, 25)
    pixels = mean_shift.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    cv2.setRNGSeed(10)
    _, labels, colors = cv2.kmeans(pixels, 8, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(gray.shape)
    segments = colors[labels].astype(np.uint8)

    source = image.astype(np.float32)
    smoothed = source.copy()
    for label in range(len(colors)):
        region = labels == label
        mask = region.astype(np.float32)
        values = cv2.GaussianBlur(source * mask[:, :, None], (15, 15), 4)
        weights = cv2.GaussianBlur(mask, (15, 15), 4)
        smoothed[region] = values[region] / weights[region, None]

    alpha = cv2.GaussianBlur(cv2.dilate(edges, None), (7, 7), 0)
    alpha = np.maximum(alpha, edges)[:, :, None] / 255
    result = np.rint(alpha * source + (1 - alpha) * smoothed).clip(0, 255).astype(np.uint8)
    save_grid("exercise_14.png", [
        ("Исходное изображение", image), ("Основные границы", edges),
        ("Цветовые области", segments), ("После альфа-смешивания", result),
    ], 2)


def line_response(gray, angle):
    y, x = np.mgrid[-20:21, -20:21]
    angle = np.deg2rad(angle)
    along = x * np.cos(angle) - y * np.sin(angle)
    across = np.abs(x * np.sin(angle) + y * np.cos(angle))
    center = (np.abs(along) <= 19) & (across <= 1.5)
    sides = (np.abs(along) <= 19) & (across >= 3) & (across <= 5)
    kernel = center / center.sum() - sides / sides.sum()
    return np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel.astype(np.float32)))


def exercise_15(image, gray):
    response = line_response(gray, 60)
    neighbors = np.maximum(line_response(gray, 45), line_response(gray, 75))
    hits = (response >= 15) & (response > 1.2 * neighbors)
    result = image.copy()
    result[hits] = (20, 30, 255)
    save_grid("exercise_15.png", [
        ("Исходное изображение", image),
        ("Отклик ядра 60°", np.clip(response * 4, 0, 255)),
        ("Выделенные участки", result),
    ], 3)


def exercise_16(image):
    source = image.astype(np.float32)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    row = np.array([[1, 2, 1]], dtype=np.float32) / 4
    direct = cv2.filter2D(source, -1, kernel)
    horizontal = cv2.filter2D(source, -1, row)
    separated = cv2.filter2D(horizontal, -1, row.T)
    difference = np.abs(direct - separated).max()
    save_grid("exercise_16.png", [
        ("Исходное изображение", image),
        ("Ядро 3 × 3", np.rint(direct).astype(np.uint8)),
        (f"Строка и столбец\nМакс. разность: {difference:g}", np.rint(separated).astype(np.uint8)),
    ], 3)


def exercise_18(target):
    images = []
    for size in (3, 5, 9, 13):
        for dx, dy, name in ((1, 0, "∂I/∂x"), (0, 1, "∂I/∂y"),
                             (2, 0, "∂²I/∂x²"), (0, 2, "∂²I/∂y²")):
            kx, ky = cv2.getDerivKernels(dx, dy, size, normalize=True)
            derivative = cv2.sepFilter2D(target, cv2.CV_32F, kx, ky)
            limit = np.abs(derivative).max()
            visible = 127.5 + derivative * (127.5 / (limit or 1))
            images.append((f"{size} × {size}, {name}", visible))
    save_grid("exercise_18.png", images, 4)


def main():
    RESULTS.mkdir(exist_ok=True)
    scene = FOLDER / "scene.jpg"
    if scene.is_file():
        image = read_image(scene)
        detailed = image
    else:
        photos = FOLDER.parent / "task2"
        image = read_image(photos / "i2.jpg")
        detailed = read_image(photos / "i3.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    exercise_12(detailed)
    exercise_13(gray)
    exercise_14(image, gray)
    exercise_15(image, gray)
    exercise_16(image)

    target = FOLDER / "target.png"
    if target.is_file():
        exercise_18(read_image(target, cv2.IMREAD_GRAYSCALE))


if __name__ == "__main__":
    main()
