from dataclasses import asdict
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from nicegui import ui
from io import BytesIO
import zipfile
import tempfile
import os
import datetime
import asyncio


def update_plot_image(fig, image_element):
    fig_buffer = BytesIO()
    fig.savefig(fig_buffer, format='png')
    fig_buffer.seek(0)
    # plt.close(fig)
    fig.clf()
    plt.close(fig)
    img = Image.open(fig_buffer)
    image_element.set_source(img)



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

    # Create a real temp file for the ZIP archive
    fd, zip_path = tempfile.mkstemp(suffix='.zip')
    os.close(fd)  # Close the file descriptor, we only need the path

    try:
        with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for name, plotter in plotters.items():
                # 🧠 Your custom plotter logic
                fig = plotter.plot(
                    **asdict(configs[plotter.name]),
                    ui=True
                )
                img_buf = BytesIO()
                fig.savefig(img_buf, format=save_format, bbox_inches='tight')
                plt.close(fig)
                img_buf.seek(0)

                # Add image to the zip in memory
                c_name = name.replace(":","_")
                zipf.writestr(f'graph_{c_name}.{save_format}', img_buf.read())

        # 🎉 Download trigger
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'all_graphs_{now}.{save_format}.zip'
        ui.download(zip_path, filename=filename)

        # ✅ Notify user
        ui.notify(f'📦 {len(plotters)} grafů exportováno jako {save_format.upper()}')

        # ⏳ Optional: cleanup
        if auto_cleanup:
            async def delayed_delete(path: str, delay: float):
                await asyncio.sleep(delay)
                try:
                    os.remove(path)
                except Exception as e:
                    print(f'Warning: Failed to delete {path}: {e}')

            asyncio.create_task(delayed_delete(zip_path, cleanup_delay))

    except Exception as e:
        ui.notify(f'❌ Chyba při exportu: {e}', type='negative')
