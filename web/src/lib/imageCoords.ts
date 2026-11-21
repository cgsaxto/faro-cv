export interface RenderedImageBox {
  offsetX: number;
  offsetY: number;
  renderedW: number;
  renderedH: number;
}

export interface PixelCoordinateHit {
  x: number;
  y: number;
  inside: boolean;
}

/**
 * Compute the exact rendered pixel rectangle of an `object-contain` image inside its container.
 * Eliminates coordinate sampling drift whenever the container aspect ratio differs from the image.
 */
export function computeObjectContainBox(
  containerW: number,
  containerH: number,
  naturalW: number,
  naturalH: number,
): RenderedImageBox {
  if (containerW <= 0 || containerH <= 0 || naturalW <= 0 || naturalH <= 0) {
    return { offsetX: 0, offsetY: 0, renderedW: 0, renderedH: 0 };
  }
  const scale = Math.min(containerW / naturalW, containerH / naturalH);
  const renderedW = naturalW * scale;
  const renderedH = naturalH * scale;
  return {
    offsetX: (containerW - renderedW) / 2,
    offsetY: (containerH - renderedH) / 2,
    renderedW,
    renderedH,
  };
}

/**
 * Map a pointer position `(clientX, clientY)` inside an `object-contain` image box
 * to exact integer pixel coordinates `[0..naturalW-1, 0..naturalH-1]`.
 */
export function clientPointToImagePixel(
  clientX: number,
  clientY: number,
  containerRect: { left: number; top: number; width: number; height: number },
  naturalW: number,
  naturalH: number,
): PixelCoordinateHit | null {
  const box = computeObjectContainBox(
    containerRect.width,
    containerRect.height,
    naturalW,
    naturalH,
  );
  if (box.renderedW <= 0 || box.renderedH <= 0) return null;

  const relX = clientX - containerRect.left - box.offsetX;
  const relY = clientY - containerRect.top - box.offsetY;
  const inside = relX >= 0 && relX <= box.renderedW && relY >= 0 && relY <= box.renderedH;

  const nx = Math.max(0, Math.min(0.999999, relX / box.renderedW));
  const ny = Math.max(0, Math.min(0.999999, relY / box.renderedH));

  return {
    x: Math.floor(nx * naturalW),
    y: Math.floor(ny * naturalH),
    inside,
  };
}

/**
 * Convert an integer image pixel center `(x + 0.5, y + 0.5)` into container-relative CSS percentages,
 * accounting for any `object-contain` letterbox padding.
 */
export function imagePixelToContainerPercent(
  x: number,
  y: number,
  containerW: number,
  containerH: number,
  naturalW: number,
  naturalH: number,
): { leftPct: number; topPct: number } {
  const box = computeObjectContainBox(containerW, containerH, naturalW, naturalH);
  if (containerW <= 0 || containerH <= 0 || box.renderedW <= 0 || box.renderedH <= 0) {
    return {
      leftPct: ((x + 0.5) / Math.max(naturalW, 1)) * 100,
      topPct: ((y + 0.5) / Math.max(naturalH, 1)) * 100,
    };
  }
  const px = box.offsetX + ((x + 0.5) / naturalW) * box.renderedW;
  const py = box.offsetY + ((y + 0.5) / naturalH) * box.renderedH;
  return {
    leftPct: (px / containerW) * 100,
    topPct: (py / containerH) * 100,
  };
}
