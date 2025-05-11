# constructs images from the shap and lime values from the ferret_xai json files
# t45

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from matplotlib.cm import get_cmap
from matplotlib import rcParams
import torch
import matplotlib
# use agg backend
matplotlib.use('agg')

def save_highlighted_words_as_png_devnagri(words, intensities, output_file, font_name, title=""):
    """
    Highlights a list of words in Devanagari script with background colors determined by their intensities 
    using the Inferno colormap, and saves the output to a PNG file.

    Args:
        words (list of str): List of words or strings in Devanagari script.
        intensities (list of float): List of floating-point numbers representing the intensity of each word.
        output_file (str): Path to save the PNG file.

    Returns:
        None: Saves the PNG file to the specified path.
    """
    if len(words) != len(intensities):
        print(f"\n{title}\n\tThe length of words and intensities are not the same. So skipping.")
        return
    
    # Normalize the intensities for colormap scaling
    norm = Normalize(vmin=min(intensities), vmax=max(intensities))
    cmap = matplotlib.colormaps.get_cmap('bwr')

    # Use a Devanagari-compatible font
    rcParams['font.family'] = [font_name,"Noto Sans"]  # Replace with a font installed on your system
    rcParams['font.size'] = 12

    # Prepare figure and axis
    fig_width, fig_height = 16, 4  # Aspect ratio 16:3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # ax.axis('off')  # Turn off axes
    # set title
    ax.set_title(title, fontsize=12)

    # Adjust field of view
    # ax.set_xlim(0, 1.0)  # Increase horizontal field of view
    # ax.set_ylim(0, -0.2)  # Increase vertical field of view

    # Define layout parameters
    x_pos, y_pos = 0.01, 0.5  # Start positions
    max_width = 0.98  # Maximum width before wrapping
    padding_x, padding_y = 0.01, 0.05  # Padding for rectangles

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    text_height = len(words)*0.0017 #0.5  # Fixed height

    for word, intensity in zip(words, intensities):
        # Get background color based on intensity
        rgba = cmap(norm(intensity))
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "black" if brightness > 0.5 else "white"

        # Estimate word width (proportional to length)
        text_width = len(word) * 0.01  # Adjust scaling factor for font size

        # Wrap to the next line if exceeding max width
        if x_pos + text_width + padding_x > max_width:
            x_pos = 0.01
            y_pos -= text_height + padding_y

        # Update bounding box
        min_x = min(min_x, x_pos)
        max_x = max(max_x, x_pos + text_width)
        min_y = min(min_y, y_pos - text_height / 2)
        max_y = max(max_y, y_pos + text_height / 2)

        # Draw rectangle (highlight)
        ax.add_patch(plt.Rectangle(
            (x_pos, y_pos - text_height / 2),
            text_width,
            text_height,
            color= hex_color #'#BDBDBD'
        ))
        # print(word, x_pos, y_pos - text_height / 2, text_width, text_height)
        # print(y_pos, text_height)

        # Add text in Devanagari
        ax.text(
            x_pos + text_width / 2, y_pos,
            word,
            color=text_color,
            ha='center',
            va='center',
            fontsize=12
        )

        # Move x_pos for the next word
        x_pos += text_width + padding_x

    # Add margins and set limits for zooming
    margin = 0.10  # Add 5% margin around the bounding box
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # plt.show()
    # Save as PNG
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)

# Example usage with Devanagari script
# words = hindi_counting = [
#      "एक", "दो", "तीन", "चार", "पाँच", "छह", "सातआठनौ"
# ]*60
# # "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस", "इकतालीस", "बयालीस", "तैतालीस", "चौंतालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवालीस", "पचपन", "छप्पन", "सत्तावन", "अठावन", "उनसठ", "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर", "इकहत्तर", "बहत्तर", "तिरेहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सत्तहत्तर", "अठहत्तर", "उनासी", "अस्सी", "इक्यासी"
# import numpy as np
# intensities = np.random.rand(len(words)) #[0.5]*len(words)
# save_highlighted_words_as_png_devnagri(words, intensities, "highlighted_words_devnagri.png", "Noto Sans Devanagari")

import json, os
from tqdm import tqdm
import glob, re
script_dict = {'hindi': 'Noto Sans Devanagari', 'urdu':'Noto Nastaliq Urdu', 'tamil':'Noto Sans Tamil', 'telugu':'Noto Sans Telugu'}
pref = 'data/'

fname_list = glob.glob(pref+'ferret_xai/*.json')

for fi,fname in enumerate(fname_list):
    folder_name = fname.split('/')[-1].split('.')[0]
    os.makedirs(f"{pref}ferret_xai/images/{folder_name}", exist_ok=True)
    with open(fname, 'r') as f:
        data = json.load(f)
    lime_count, shap_count = 1, 1
    pbar = tqdm(enumerate(data[list(data.keys())[0]]), ncols=100, total=len(data[list(data.keys())[0]]))
    for qasi,qas in pbar:
        for tdi,text_dict in enumerate(qas):
            words = text_dict['Token']
            shap_intensities = text_dict['Partition SHAP']
            url = text_dict['url']
            language = re.search(r'bbc\..*?\/\w+',url)[0].split('/')[1]
            pbar.set_description(f"Processing {folder_name} {language} file {fi+1}/{len(fname_list)} qas {qasi+1}/{len(data[list(data.keys())[0]])} tdi {tdi+1}/{len(qas)}")
            script = script_dict[language]
            # print('Label:', text_dict['label'])
            # print('Prediction:', 1.0 if text_dict['prediction'][0][0] > text_dict['prediction'][0][1] else 0.0)
            logit = text_dict['prediction'][0]
            logit = round(torch.softmax(torch.tensor(logit), dim=0).tolist()[0],3)
            title = f"SHAP {language} {fname} Label {text_dict['label']} Prediction {logit} qasi {qasi} tdi {tdi}"
            if not os.path.exists(f"{pref}ferret_xai/images/{folder_name}/shap_{shap_count}.png"):
                save_highlighted_words_as_png_devnagri(words, shap_intensities, f"{pref}ferret_xai/images/{folder_name}/shap_{shap_count}.png", script, title)
            shap_count += 1
            # print('English '*10)
            words = data['english'][qasi][tdi]['Token']
            shap_intensities = data['english'][qasi][tdi]['Partition SHAP']
            # print('Label:', data['english'][qasi][tdi]['label'])
            # print('Prediction:', 1.0 if data['english'][qasi][tdi]['prediction'][0][0] > data['english'][qasi][tdi]['prediction'][0][1] else 0.0)
            logit_en = data['english'][qasi][tdi]['prediction'][0]
            logit_en = round(torch.softmax(torch.tensor(logit_en), dim=0).tolist()[0],3)
            title_en = f"SHAP English {fname} Label {data['english'][qasi][tdi]['label']} Prediction {logit_en} qasi {qasi} tdi {tdi}"
            if not os.path.exists(f"{pref}ferret_xai/images/{folder_name}/shap_{shap_count}.png"):
                save_highlighted_words_as_png_devnagri(words, shap_intensities, f"{pref}ferret_xai/images/{folder_name}/shap_{shap_count}.png", "Noto Sans", title_en)
            shap_count += 1

            # print('\n',logit, logit_en)

            # print('LIME '*20)
            title = f"LIME {language} {fname} Label {text_dict['label']} Prediction {logit} qasi {qasi} tdi {tdi}"
            words = text_dict['Token']
            lime_intensities = text_dict['LIME']
            if not os.path.exists(f"{pref}ferret_xai/images/{folder_name}/lime_{lime_count}.png"):
                save_highlighted_words_as_png_devnagri(words, lime_intensities, f"{pref}ferret_xai/images/{folder_name}/lime_{lime_count}.png", script, title)
            lime_count += 1
            # print('English '*10)
            words = data['english'][qasi][tdi]['Token']
            lime_intensities = data['english'][qasi][tdi]['LIME']
            title_en = f"LIME English {fname} Label {data['english'][qasi][tdi]['label']} Prediction {logit_en} qasi {qasi} tdi {tdi}"
            if not os.path.exists(f"{pref}ferret_xai/images/{folder_name}/lime_{lime_count}.png"):
                save_highlighted_words_as_png_devnagri(words, lime_intensities, f"{pref}ferret_xai/images/{folder_name}/lime_{lime_count}.png", "Noto Sans", title_en)
            lime_count += 1
            # print('\n',logit, logit_en)
            # input('wait')

