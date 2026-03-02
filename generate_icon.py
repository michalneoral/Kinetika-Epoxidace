from PIL import Image

# Load base image (1024x1024 transparent PNG)
img = Image.open("icon.png")

# --- Windows ICO (app icon) ---
img.save(
    "build/icon.ico",
    sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]
)

# --- Browser favicon ICO (tab icon) ---
# Doporučené velikosti pro favicon: 16/32, volitelně 48/64.
img.save(
    "static_favicon.ico",
    sizes=[(16,16), (32,32), (48,48), (64,64)]
)

# --- macOS ICNS ---
# Pillow 9.2+ supports icns export
img.save("build/icon.icns")
