import threading
import time
from app.version import __version__
from app.updater import maybe_update_in_background

def main():
    print(f"MyApp v{__version__} starting…")

    # fire-and-forget update check (won't block startup)
    threading.Thread(target=maybe_update_in_background, daemon=True).start()

    # --- your actual app here (CLI or GUI loop) ---
    # Example placeholder loop:
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
