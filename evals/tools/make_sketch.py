"""Draw the sketch of the eval case `sketch-no-size`: a wall hook in side view.

The drawing has proportions and no size: no number, no dimension line and no known object
(no coin, no screw with a mark). A model can read the shape from it, and it has to ask for
the scale.

Usage (in the repo root):
    python3 evals/tools/make_sketch.py evals/sketch-no-size/files/sketch.png

It needs Pillow. The same script gives the same file.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 640, 480
INK, PAPER = 0, 255
LINE = 5

# The profile of the hook (side view): the wall plate on the left, the arm that goes out to the right,
# the lip at the end of the arm. The points go around the shape once.
PROFILE = [(80, 110), (125, 110), (125, 255), (430, 255), (430, 190), (485, 190), (485, 310),
           (125, 310), (125, 370), (80, 370)]


def draw(out: Path) -> None:
    img = Image.new("L", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    # the wall: one vertical line with short hatch marks on the far side
    d.line([(80, 40), (80, 440)], fill=INK, width=LINE)
    for y in range(50, 440, 24):
        d.line([(80, y + 18), (58, y)], fill=INK, width=2)
    # the hook
    d.line(PROFILE + [PROFILE[0]], fill=INK, width=LINE, joint="curve")
    # two screws through the plate into the wall, as dashed lines with a head on the outer face
    for y in (150, 330):
        for x in range(62, 125, 12):
            d.line([(x, y), (x + 6, y)], fill=INK, width=2)
        d.line([(132, y - 10), (132, y + 10)], fill=INK, width=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)


if __name__ == "__main__":
    draw(Path(sys.argv[1]))
