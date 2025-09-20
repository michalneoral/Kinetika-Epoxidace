from PIL import Image

# Load base image (1024x1024 transparent PNG)
img = Image.open("build/icon.png")

# --- Windows ICO ---
img.save("build/icon.ico", sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])

# --- macOS ICNS ---
# Pillow 9.2+ supports icns export
img.save("build/icon.icns")
