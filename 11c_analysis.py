
# make an analysis of lime and shape scores from ferrit_xai json files
# based on my hypothesis, "Higher intensity is given to all words when model predicts values close to 1"
# t45_v2

import json, glob, re, os, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
# use agg backend
matplotlib.use('agg')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE+4)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE+5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE+5)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE+5)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ml = glob.glob('data/ferret_xai/*.json')

# intensity_distribution = {'shap':{'hindi':{}, 'urdu':{}, 'tamil':{}, 'telugu':{}, 'english':{}, 'all':{}}, 'lime':{'hindi':{}, 'urdu':{}, 'tamil':{}, 'telugu':{}, 'english':{}, 'all':{}}} 

# print(intensity_distribution['shap']['hindi'])

val_list = [0.5,0.6,0.7,0.8,0.9]

def mymap(fval):
    if fval < 0.1:
        return 10
    elif fval < 0.2:
        return 20
    elif fval < 0.3:
        return 30
    elif fval < 0.4:
        return 40
    elif fval < 0.5:
        return 50
    elif fval < 0.6:
        return 60
    elif fval < 0.7:
        return 70
    elif fval < 0.8:
        return 80
    elif fval < 0.9:
        return 90
    else:
        return 100

def Normalize(lst):
    '''
    lst can have negative values
    '''
    mx = max(lst)
    mn = min(lst)
    if mx == mn:
        return [0.5]*len(lst)
    return [(i-mn)/(mx-mn) for i in lst]

def plot_multiple_dictionaries(dict_list):
    """
    Plot multiple dictionaries with float keys and values on the same figure.
    Each dictionary will have a unique line style and color.
    """
    if not dict_list:
        raise ValueError("The list of dictionaries is empty.")
    
    plt.figure(figsize=(10, 6))

    for idx, data in enumerate(dict_list):
        if not data:
            raise ValueError(f"Dictionary at index {idx} is empty.")
        
        # Sort each dictionary by keys
        sorted_data = dict(sorted(data.items()))
        x = list(sorted_data.keys())
        y = [t*100 for t in list(sorted_data.values())]

        # Plot with a unique label
        plt.plot(x, y, marker='o', linestyle='-', label=f'threshold={val_list[idx]}')
        # make x axis labels
        x2 = [str(i-10)+'-'+str(i) for i in range(10, 101, 10)]
        plt.xticks(x, x2)

    # Customizing the plot
    plt.xlabel('Logit buckets')
    plt.ylabel('% tokens whose relevance exceeded threshold')
    # plt.title('Logit Buckets vs % tokens according to threshold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # plt.show()
    # Save the plot as an image file
    plt.savefig('data/ferret_xai/images/thresholds_lime.svg', format='svg', dpi=300)
    plt.close()  # Close the plot to free memory


dict_list = []

for threshold in val_list:
    intensity_distribution = {'shap':{'hindi':{}, 'urdu':{}, 'tamil':{}, 'telugu':{}, 'english':{}, 'all':{}}, 'lime':{'hindi':{}, 'urdu':{}, 'tamil':{}, 'telugu':{}, 'english':{}, 'all':{}}} 
    for k in intensity_distribution.keys():
        for l in intensity_distribution[k].keys():
            for i in range(10, 101, 10):
                intensity_distribution[k][l][i] = []
    for fname in tqdm(ml, ncols=100):
        with open(fname, 'r') as f:
            data = json.load(f)
        for language in data.keys():
            for qasi, qas in enumerate(data[language]):
                for tdi, text_dict in enumerate(qas):
                    logit = text_dict['prediction'][0]
                    logit = round(torch.softmax(torch.tensor(logit), dim=0).tolist()[0],3)
                    token = text_dict['Token']
                    if len(token) != len(text_dict['Partition SHAP']):
                        continue
                    shap_intensities = Normalize(text_dict['Partition SHAP'])
                    lime_intensities = Normalize(text_dict['LIME'])
                    logit_mapped = mymap(logit)
                    shap_thresholded = [1 if i > threshold else 0 for i in shap_intensities]
                    lime_thresholded = [1 if i > threshold else 0 for i in lime_intensities]
                    shap_count = sum(shap_thresholded)
                    lime_count = sum(lime_thresholded)
                    shap_percentage = shap_count/len(shap_thresholded)
                    lime_percentage = lime_count/len(lime_thresholded)
                    intensity_distribution['shap'][language][logit_mapped].append(shap_percentage)
                    intensity_distribution['lime'][language][logit_mapped].append(lime_percentage)
                    intensity_distribution['shap']['all'][logit_mapped].append(shap_percentage)
                    intensity_distribution['lime']['all'][logit_mapped].append(lime_percentage)

    # show average
    for k in intensity_distribution.keys():
        for l in intensity_distribution[k].keys():
            for i in range(10, 101, 10):
                if len(intensity_distribution[k][l][i]) > 0:
                    intensity_distribution[k][l][i] = sum(intensity_distribution[k][l][i])/len(intensity_distribution[k][l][i])
                else:
                    intensity_distribution[k][l][i] = 0

    dict_list.append(intensity_distribution['lime']['all'])

plot_multiple_dictionaries(dict_list)

# pretty print
# print(json.dumps(intensity_distribution, indent=2))





    
    

