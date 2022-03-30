# Compute the size of pixel

```
python compute_area.py --input {image directory} --meta {location of meta data (or save directory)}
```
meta data is maintained in json file. <br>
Image file name should consist of `{name of camera}-{numbering}` <br>
By default, separator is `-`, but you can change it in `SEPARATOR`

## Current version supports only if there are circles of the same size in images.

Hough transformation algorithm searches for known size of circles. `ACTUAL_DIM` (diameter of circle, unit: cm)

<div align="center">
    <img src = "demo/pixelratio_fit.png" width="700"/> <br>
    <figcaption align = "center"><b>Fig.1 - Fitting circles </b></figcaption>
</div>
<br />

In case no appropriate representative circles, the image is skipped.

<div align="center">
    <img src = "demo/pixelratio_notfit.png" width="400"/> <br>
    <figcaption align = "center"><b>Fig.2 - Fail case </b></figcaption>
</div>
<br />