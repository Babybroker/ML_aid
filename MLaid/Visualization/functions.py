from matplotlib.pyplot import savefig
from datetime import datetime as dt


def save_func(save_visual, timestamp, folder_name, filename):
    save_timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    if save_visual:
        if timestamp is None:
            timestamp = save_timestamp
        filename = f'{SAVE_LOCATION_VISUALS}/figs/model_{timestamp}.png' if folder_name is None \
            else folder_name + f'{filename}_{timestamp}.png'
        savefig(filename)
