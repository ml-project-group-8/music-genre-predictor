import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('final.csv')
colnames = list(data)
label = ['rnb', 'hiphop', 'country', 'classical', 'edm_dance', 'rock']

rnb_data = data.loc[data['Genre'] == 'rnb']
hiphop_data = data.loc[data['Genre'] == 'hiphop']
country_data = data.loc[data['Genre'] == 'country']
classical_data = data.loc[data['Genre'] == 'classical']
edm_data = data.loc[data['Genre'] == 'edm_dance']
rock_data = data.loc[data['Genre'] == 'rock']

energy_data = []
for i in [rnb_data, hiphop_data, country_data, classical_data, edm_data, rock_data]:
    energy_data.append(np.average(i['energy']*100))

accousticness_data = []
for i in [rnb_data, hiphop_data, country_data, classical_data, edm_data, rock_data]:
    accousticness_data.append(np.average(i['acousticness']*100))

def energy_bar_x():
    """
    generates bar plot for the 'energy' feature of each genre
    data used is the average of all the energy scores
    """
    index = np.arange(len(label))
    plt.bar(index, energy_data, color=['magenta', 'red', 'green', 'blue', 'black', 'purple'])
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('Energy Score', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Energy Score for Different Genres')
    plt.show()

def accousticness_bar_x():
    """
    generates bar plot for the 'accousticness' feature of each genre
    data used is the average of all the accousticness scores
    """
    index = np.arange(len(label))
    plt.bar(index, accousticness_data, color=['magenta', 'red', 'green', 'blue', 'black', 'purple'])
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('Accoustic Score', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Accoustic Score for Different Genres')
    plt.show()

energy_bar_x()
accousticness_bar_x()
