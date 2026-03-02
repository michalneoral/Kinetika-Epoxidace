from dataclasses import asdict
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib
from nicegui import ui
from io import BytesIO
import zipfile
import tempfile
import os
import datetime
import asyncio

# NOTE (PyInstaller): PDF/SVG export requires optional Matplotlib backends.
# In frozen builds these backends may not be collected automatically unless
# they are imported somewhere in the code. Import them best-effort here so
# export works in packaged EXE.
try:  # pragma: no cover
    import matplotlib.backends.backend_pdf  # noqa: F401
    import matplotlib.backends.backend_svg  # noqa: F401
except Exception:
    pass


def update_plot_image(fig, image_element):
    """Render matplotlib figure as PNG and set it as data URL on a NiceGUI (interactive_)image.
    This avoids temporary files (important on Windows) and keeps click coordinates stable.
    """
    import base64
    fig_buffer = BytesIO()
    fig.savefig(fig_buffer, format='png')
    fig_buffer.seek(0)
    data = fig_buffer.getvalue()
    b64 = base64.b64encode(data).decode('ascii')
    image_element.set_source(f'data:image/png;base64,{b64}')
    fig.clf()
    plt.close(fig)

async def export_all_figures_as_zip_old(plotters, configs, save_format='png'):
    # Use temp dir to hold image files and zip
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []

        for name, plotter in plotters.items():
            fig = plotter.plot(
                **asdict(configs[plotter.name]),
                ui=True
            )

            path = os.path.join(tmpdir, f'graph_{name.replace(":","_")}.{save_format}')
            fig.savefig(path, bbox_inches='tight')
            image_paths.append(path)
            plt.close(fig)

        # Create zip file
        zip_path = os.path.join(tmpdir, 'all_graphs.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for img_path in image_paths:
                arcname = os.path.basename(img_path)
                zipf.write(img_path, arcname)

        # Provide file for download
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f'graphs_{now}.zip'
        ui.download(zip_path, filename=download_name)

async def export_all_figures_as_zip(plotters: dict,
                                    configs: dict,
                                    save_format: str = 'png',
                                    auto_cleanup: bool = True,
                                    cleanup_delay: float = 15.0):
    assert save_format in ('png', 'pdf', 'svg'), f"Unsupported format: {save_format}"

    # NOTE: Use in-memory bytes to avoid temp-file issues on Windows.
    try:
        out = BytesIO()
        with zipfile.ZipFile(out, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for name, plotter in plotters.items():
                cfg = configs[plotter.name]
                kwargs = cfg.to_kwargs() if hasattr(cfg, 'to_kwargs') else asdict(cfg)
                fig = plotter.plot(ui=True, **kwargs)
                img_buf = BytesIO()
                fig.savefig(img_buf, format=save_format, bbox_inches='tight')
                plt.close(fig)
                img_buf.seek(0)
                c_name = name.replace(':', '_')
                zipf.writestr(f'graph_{c_name}.{save_format}', img_buf.read())

        zip_bytes = out.getvalue()

        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'all_graphs_{now}.{save_format}.zip'
        ui.download(zip_bytes, filename=filename)
        ui.notify(f'📦 {len(plotters)} grafů exportováno jako {save_format.upper()}')
    except Exception as e:
        ui.notify(f'❌ Chyba při exportu: {e}', type='negative')
