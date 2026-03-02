"""Application identity.

APP_NAME is a stable, filesystem-safe identifier (used for platformdirs paths,
app ping payloads, etc.). Do not change lightly, otherwise users end up with a
new data directory.

APP_DISPLAY_NAME is what the user sees in the UI / installer.
"""

APP_NAME = "FAME_EPO_Manager"
APP_DISPLAY_NAME = "FAME EPO Manažer"
APP_AUTHOR = "FAME_EPO_MN_KD_UPCE"  # změň dle potřeby

# GitHub updater defaults (hardcoded for this project)
UPDATE_REPO_URL = "https://github.com/michalneoral/Kinetika-Epoxidace"
UPDATE_OWNER = "michalneoral"
UPDATE_REPO = "Kinetika-Epoxidace"
