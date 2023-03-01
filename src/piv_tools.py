import numpy as np
from scipy.fft import rfft2, irfft2, fftshift
from numpy import log
import pathlib
import matplotlib.pyplot as plt
import openpiv.tools as pivtools
import openpiv.scaling as pivscaling
import openpiv.filters as pivfilters
import openpiv.validation as pivval
from openpiv.tools import imread
from openpiv.pyprocess import (
    extended_search_area_piv,
    normalize_intensity,
    get_coordinates,
)
from PIL import Image

# C:\Users\adamk\Projects\slope-finder\img\piv\4238_1224.tif
frame_a = Image.open("img/piv/4238_1224.tif")
frame_b = Image.open("img/piv/4238_1228.tif")

# test
frame_a = np.array(frame_a)
frame_b = np.array(frame_b)

frame_a = normalize_intensity(frame_a)
frame_b = normalize_intensity(frame_b)

window_size = 32
overlap = 16
dt = 1.0
search_area_size = 32
correlation_method = "linear"
subpixel_method = "gaussian"
sig2noise_method = "peak2peak"

# vel2 = extended_search_area_piv(
#     frame_a,
#     frame_b,
#     window_size=window_size,
#     search_area_size=search_area_size,
#     overlap=overlap,
#     dt=dt,
#     correlation_method="circular",
#     subpixel_method=subpixel_method,
#     sig2noise_method=sig2noise_method,
# )


u0, v0, sig2noise = extended_search_area_piv(
    frame_a,
    frame_b,
    window_size=window_size,
    search_area_size=search_area_size,
    overlap=overlap,
    dt=dt,
    correlation_method=correlation_method,
    normalized_correlation=True,
    subpixel_method=subpixel_method,
    sig2noise_method=sig2noise_method,
)

# print(u0[100:110])

x, y = get_coordinates(
    image_size=frame_a.shape,
    search_area_size=search_area_size,
    overlap=overlap,
)

plt.hist(sig2noise.flatten())

invalid_mask = pivval.sig2noise_val(
    sig2noise,
    threshold=0.01,
)


u2, v2 = pivfilters.replace_outliers(
    u0,
    v0,
    invalid_mask,
    method="localmean",
    max_iter=1,
    kernel_size=3,
)


x, y, u3, v3 = pivscaling.uniform(
    x,
    y,
    u2,
    v2,
    scaling_factor=4.75,  # 96.52 pixels/millimeter 238/5cm =
)

# 0,0 shall be bottom left, positive rotation rate is counterclockwise
x, y, u3, v3 = pivtools.transform_coordinates(x, y, u2, v2)

pivtools.save("config/exp1_001.txt", x, y, u3, v3, invalid_mask)

vorticity = v3 / x + u3 / y

# vorticity_filtered = vorticity
print(f"{np.shape(u3)=}")
print(f"{np.shape(v3)=}")

plt.figure(figsize=(20, 20))
# cp = plt.contourf(u3, v3, vorticity)
# cb = plt.colorbar(cp)
# plt.quiver(u3, v3, scale=50, alpha=0.5, cmap="winter_r")
plt.scatter(u3[0], u3[1])
plt.show()


# Vectors with quiverplt.imshow(phi, extent=(0,10,0,10), interpolation='none', origin='lower')plt.quiver(x, y, np.gradient(phi)[0], np.gradient(phi)[1])

fig, ax = plt.subplots(figsize=(20, 8))

pivtools.display_vector_field(
    pathlib.Path("config/exp1_001.txt"),
    ax=ax,
    scaling_factor=4.75,
    scale=50,  # scale defines here the arrow length
    width=0.0035,  # width is the thickness of the arrow
    on_img=True,  # overlay on the image
    image_name="img/piv/4238_1224.tif",
    show_invalid=False,
)


# dy, dx = np.gradient(p, np.diff(y[:2, 0]), np.diff(x[0, :2]))
# skip = (slice(None, None, 3), slice(None, None, 3))

# fig2, ax2 = plt.subplots()
# im = ax2.imshow(frame_a, extent=[x.min(), x.max(), y.min(), y.max()])
# ax2.quiver(x[skip], y[skip], dx[skip], dy[skip])

# fig2.colorbar(frame_a)
# ax2.set(aspect=1, title="Quiver Plot")
# plt.show()
