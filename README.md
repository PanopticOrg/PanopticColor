# Panoptic Colors 🌈

Simple plugin that computes the mean color of images and stores it in properties, in order to filter, sort, group and lay out images by color.

![image](./image.png)


## Usage

The plugin stores, for each image, these number properties, in a `Colors` property group:

| Property | Component | Range |
|---|---|---|
| `color_R`, `color_G`, `color_B` | red, green, blue | 0 – 255 |
| `color_H` | hue | 0 – 359 (degrees) |
| `color_S` | saturation | 0 – 100 |
| `color_V` | value | 0 – 100 |
| `color_L` | perceived luminance | 0 – 100 |

It adds three functions:

- **compute_colors** (execute button): computes the colors of the selected images in a background task.
- **cluster_by_colors** (group): K-means clustering of the images by color, in the RGB, HSV or both color spaces.
- **color_map** (map view): creates a spatial view placing images by two color components, e.g. Hue on the horizontal axis and Saturation on the vertical axis. With the `radial` option, the first component gives the angle and the second the distance to the center: Hue / Saturation then draws a color wheel. `reverse_x` / `reverse_y` reverse an axis, e.g. `radial` + `reverse_y` puts the most saturated images at the center.

`cluster_by_colors` and `color_map` compute the missing colors themselves, so `compute_colors` is not required before using them.
