import json

def remove_painted_pixels(file, n):
    with open(file, encoding='utf-8') as f:
        d = json.load(f)
    w = d['imageData']['width']
    h = d['imageData']['height']
    d['progress']['paintedPixels'] = n
    y_last = n // w
    x_last = n % w
    d['progress']['lastPosition']['x'] = x_last
    d['progress']['lastPosition']['y'] = y_last
    first_remaining = d['remainingPixels'][0]
    to_be_removed = n - (first_remaining['imageY'] * w + first_remaining['imageX'])
    d['remainingPixels'] = d['remainingPixels'][to_be_removed:]
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    remove_painted_pixels(
        r'C:\Users\SceneryMC\Downloads\wplace_progress_102892926_p0_converted.png_2025-09-06T15-22-46.json',
        35358)