from typing import Union, Tuple
import matplotlib.colors as mcolors

def convert_color(
    color: Union[str, Tuple[int, ...], Tuple[float, ...]],
    format: str = 'rgba'
) -> Union[str, Tuple, Tuple[Tuple, float]]:
    """
    Convert input color to various formats.
    Supported formats:
        'rgb', 'rgba', 'RGB', 'RGBA',
        'hex', 'hexa',
        'rgb-a', 'RGB-A', 'hex-a'
    """
    # First normalize to float rgba
    if isinstance(color, str) and color.startswith("#") and len(color) == 9:
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        a = int(color[7:9], 16) / 255.0
    else:
        # Normalize int tuples (0–255) to floats
        if isinstance(color, (tuple, list)):
            if all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                color = tuple(c / 255.0 for c in color)

        r, g, b, a = mcolors.to_rgba(color)

    rgb_float = (r, g, b)
    rgba_float = (r, g, b, a)
    rgb_int = tuple(int(round(c * 255)) for c in (r, g, b))
    rgba_int = (*rgb_int, int(round(a * 255)))

    print(rgba_int)

    fmt = format

    # === Two-output formats ===
    if fmt in ['rgb-a', 'rgb_a']:
        return rgb_float, a
    if fmt in ['rgb-a-int', 'RGB-A']:
        return rgb_int, int(round(a * 255))
    if fmt == 'hex-a':
        return '#{:02X}{:02X}{:02X}'.format(*rgb_int), a
    

    # === Single-output formats ===
    if fmt == 'rgb':
        return rgb_float
    if fmt == 'rgba':
        return rgba_float
    if fmt in ['rgb-int', 'RGB']:
        return rgb_int
    if fmt in ['rgba-int', 'RGBA']:
        return rgba_int
    if fmt == 'hex':
        return '#{:02X}{:02X}{:02X}'.format(*rgb_int)
    if fmt == 'hexa':
        return '#{:02X}{:02X}{:02X}{:02X}'.format(*rgb_int, int(round(a * 255)))
    
    raise ValueError(f"Unsupported format: {format}")


def test_convert_color():
    # === Float inputs
    assert convert_color((1.0, 0.5, 0.0), 'rgb') == (1.0, 0.5, 0.0)
    assert convert_color((1.0, 0.5, 0.0, 0.25), 'rgba') == (1.0, 0.5, 0.0, 0.25)

    # === Int inputs
    print(convert_color((255, 128, 0), 'RGB'))
    assert convert_color((255, 128, 0), 'RGB') == (255, 128, 0)
    assert convert_color((255, 128, 0, 64), 'RGBA') == (255, 128, 0, 64)

    # === Named color
    rgb, a = convert_color('red', 'rgb-a')
    assert rgb == (1.0, 0.0, 0.0) and a == 1.0

    # === Hex color without alpha
    assert convert_color('#FF8000', 'rgb') == (1.0, 0.5019607843137255, 0.0)
    assert convert_color('#FF8000', 'RGB') == (255, 128, 0)
    assert convert_color('#FF8000', 'hex') == '#FF8000'
    assert convert_color('#FF8000', 'hexa') == '#FF8000FF'

    # === Hex color with alpha
    assert convert_color('#FF800080', 'hexa') == '#FF800080'
    rgb, a = convert_color('#FF800080', 'rgb-a')
    assert rgb == (1.0, 0.5019607843137255, 0.0) and abs(a - 0.50196) < 1e-4

    # === Combined formats
    assert convert_color('blue', 'hex-a') == ('#0000FF', 1.0)
    assert convert_color((1.0, 0.0, 0.0, 0.5), 'RGBA') == (255, 0, 0, 128)
    assert convert_color((0.0, 1.0, 0.0, 1.0), 'RGB-A') == ((0, 255, 0), 255)

    print("All tests passed.")

if __name__ == '__main__':
    test_convert_color()