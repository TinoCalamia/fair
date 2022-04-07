import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
import random
import re
from sklearn.metrics import roc_auc_score, classification_report, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import scipy.stats as st
import tensorflow as tf

#import cv2
from deepface import DeepFace


# CREATE DF
def create_base_race_df(race, size = 1000):
    # create comparison dataframes for African faces
    directory = f"data/race_per_7000/{race}"
    evaluation_df = pd.DataFrame(columns=['original_image','original_person'])
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            if len(evaluation_df) == size:
                break
            #full file name
            filename = os.path.join(root, file)
            # person identifier
            person_folder = os.path.join(root)
            # append to original df
            evaluation_df = evaluation_df.append(pd.DataFrame({'original_image':filename,
                                                         'original_person':person_folder},index=[0]))
    return evaluation_df

def create_same_person_dataset(df):
    
    # create random same person df
    same_person = df.sample(int(len(df)/2))
    same_person['compare_person'] =same_person.original_person
    # get random image from same person
    same_person['compare_image'] = same_person.original_person.apply(lambda x: random.choice(os.listdir(x)))
    # create total image path
    same_person['compare_image'] = same_person['compare_person']+'/'+same_person['compare_image']
    same_person['is_same_person'] =True
    
    return same_person

def create_different_person_dataset(df):
    
    different_person = df
    # randomly assing images to entire dataset
    different_person['compare_image'] = shuffle(different_person.original_image)
    
    # Define regex "including until 4th slash"
    p = re.compile('.*/.*/.*/.*/')
    # cut total image path to only return person
    different_person['compare_person'] = different_person['compare_image'].apply(lambda x:
                                                                p.findall(x)[0][:-1])

    different_person['is_same_person'] = False
    # Only filter rows with different persons
    different_person = different_person[
        (different_person.compare_person!=different_person.original_person)].drop_duplicates().sample(int(len(df)/2))
    
    return different_person

def create_evaluation_dataset(df):
    # create random same person df
    same_person = create_same_person_dataset(df)
    
    # create random different person df
    different_person = create_different_person_dataset(df)

    
    # Create target variable
    evaluation = same_person.append(different_person)
    evaluation.is_same_person = evaluation.is_same_person.fillna(False)
    
    print('Target distribution is:')
    print(evaluation.is_same_person.value_counts(normalize=True))
    
    return evaluation

# PREDICTION FUNCTIONS

def normalize_distances(df, distance_column = 'distance', threshold = .5):
    
    df['distance_normalised'] = MinMaxScaler().fit_transform(np.array(df[distance_column]).reshape(-1, 1))
    df.loc[df.distance_normalised>threshold,"verified_normalised"] = False
    df.loc[df.distance_normalised<threshold,"verified_normalised"] = True

    return df

def verify_faces(df, model, performance_metrics= [roc_auc_score, accuracy_score], face_count= 100, distance_metric='cosine'):
    df['distance'] = np.nan
    df['verified'] = np.nan

    df=df.sample(frac=1)[:face_count]

    for i in range(0,len(df)):

        verification_result = DeepFace.verify(img1_path = df.iloc[i].original_image,
                          img2_path = df.iloc[i].compare_image,
                          distance_metric = distance_metric,
                           model = model,
                        enforce_detection=False)
        df.distance.iloc[i] = verification_result['distance']
        df.verified.iloc[i] = verification_result['verified']
        
    df = normalize_distances(df)
    results = {}
    results['data'] = df
    results['threshold'] = verification_result['threshold']
    
    for metric in performance_metrics:
        results[metric.__name__] = metric(df.is_same_person.astype(bool),df.verified_normalised.astype(bool))
        
    return results



# EVALUATION FUNCTIONS
def calculate_confidence_interval(model, race, metric='roc_auc_score'):
    directory = f"results/{model}/{race}"
    scores = []
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            with open(os.path.join(directory,file), "rb") as f:
                results = pickle.load(f)
            scores.append(results[metric])

    lower_ci, higher_ci = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))
    if higher_ci>1:
        higher_ci=1

    return np.mean(scores), lower_ci, higher_ci


def calculate_performance_per_threshold(model, race):
    # Final df
    performance = pd.DataFrame()
    
    # Loop to consider all files
    for j in range(1,6):
        with open(f'results/{model}/{race}/result_dict{j}.pickle', "rb") as f:
                results = pickle.load(f)
        data = results['data']
        data['distance'] = MinMaxScaler().fit_transform(np.array(data.distance).reshape(-1, 1))
        temp_performance = pd.DataFrame(columns=['threshold','score'])
        
        # Loop to define all thersholds
        for i in np.arange(0.05,1.05, 0.05):
            # Apply thresholds
            data.loc[data.distance>i,"verified"] = False
            data.loc[data.distance<i,"verified"] = True

            score = round(roc_auc_score(data.is_same_person.astype(bool), data.verified.astype(bool)),4)

            temp_performance = temp_performance.append(pd.DataFrame({'threshold':round(i,2), 'score':score},index=[0]))
            # Summarise all scores for one file
            temp_performance = temp_performance.reset_index(drop=True)
            # Append all results of all files
            performance = performance.append(temp_performance)
            
    performance = performance.groupby('threshold').score.mean().reset_index()
    return performance

# PLOTTING FUNCTIONS
def group_distances(distances, groupby):
    
    distances['distance'] = MinMaxScaler().fit_transform(np.array(distances.distance).reshape(-1, 1))
    distances = round(distances,1)

    grouped_distances = distances.groupby(groupby).original_image.count()

    grouped_distances = grouped_distances.reset_index()
    return grouped_distances

def count_distances(model, race, groupby = ['distance']):
    directory = f"results/{model}/{race}"
    distances = pd.DataFrame()
    
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            with open(os.path.join(directory,file), "rb") as f:
                results = pickle.load(f)
            data = results['data']
            distances = distances.append(pd.DataFrame(data))

    return group_distances(distances, groupby)

def calculate_distribution_difference(model, race):
    directory = f"results/{model}/{race}"
    distances = pd.DataFrame()
    
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            with open(os.path.join(directory,file), "rb") as f:
                results = pickle.load(f)
            data = results['data']
            distances = distances.append(pd.DataFrame(data))
    

    return distances.groupby("is_same_person").distance.mean().diff().loc[1]

def plot_performance_per_threshold(model):

    african_thresholds = calculate_performance_per_threshold(model = model, race = 'African')
    asian_thresholds = calculate_performance_per_threshold(model = model, race = 'Asian')
    caucasian_thresholds = calculate_performance_per_threshold(model = model, race = 'Caucasian')
    indian_thresholds = calculate_performance_per_threshold(model = model, race = 'Indian')

    f, axes = plt.subplots(2, 2, figsize=(12,12))
    sns.barplot(x='threshold', y='score', data= african_thresholds, ax=axes[0,0]).set_title('African scores')
    axes[0,0].tick_params(labelrotation=45)
    sns.barplot(x='threshold', y='score', data= asian_thresholds, ax=axes[0,1]).set_title('Asian scores')
    axes[0,1].tick_params(labelrotation=45)
    sns.barplot(x='threshold', y='score', data= caucasian_thresholds, ax=axes[1,0]).set_title('Caucasian scores')
    axes[1,0].tick_params(labelrotation=45)
    sns.barplot(x='threshold', y='score', data= indian_thresholds, ax=axes[1,1]).set_title('Indian scores')
    axes[1,1].tick_params(labelrotation=45)
    f.suptitle(f'{model} performance by threshold')
    plt.show();

def get_metric_comparision(model, metric):
    score_summary = pd.DataFrame()
    for race in ['Indian', 'Caucasian', 'African', 'Asian']:
        mean, lower_ci, higher_ci = calculate_confidence_interval(model = model, race = race, 
                                                                  metric = metric)

        score_summary = score_summary.append(pd.DataFrame({'race':race, 'mean':mean, 
                                                           'lower_ci':lower_ci, 'higher_ci':higher_ci},
                                                          index = [0]))
        score_summary.names = f"{model} - {metric}"
    return score_summary

def plot_score_distribution(model, groupby=['distance'], hue=None):

    african_distances = count_distances(model = model, race = 'African',
                                       groupby=groupby).rename(columns={'original_image':'counts'})
    asian_distances = count_distances(model = model, race = 'Asian',
                                     groupby=groupby).rename(columns={'original_image':'counts'})
    caucasian_distances = count_distances(model = model, race = 'Caucasian',
                                         groupby=groupby).rename(columns={'original_image':'counts'})
    indian_distances = count_distances(model = model, race = 'Indian',
                                      groupby=groupby).rename(columns={'original_image':'counts'})
    
    f, axes = plt.subplots(2, 2, figsize=(12,12))
    sns.barplot(x='distance', y='counts', data= african_distances, ax=axes[0,0], hue = hue).set_title('African distances')
    axes[0,0].tick_params(labelrotation=45)
    sns.barplot(x='distance', y='counts', data= asian_distances, ax=axes[0,1], hue = hue).set_title('Asian distances')
    axes[0,1].tick_params(labelrotation=45)
    sns.barplot(x='distance', y='counts', data= caucasian_distances, ax=axes[1,0], hue = hue).set_title('Caucasian distances')
    axes[1,0].tick_params(labelrotation=45)
    sns.barplot(x='distance', y='counts', data= indian_distances, ax=axes[1,1], hue = hue).set_title('Indian distances')
    axes[1,1].tick_params(labelrotation=45)
    f.suptitle(f'{model} distance distribution')
    plt.show();
    
# EVALUATION FUNCTIONS

def perform_significance_test(model, race1, race2, test_type = st.wilcoxon):
    data1 = pd.DataFrame()
    data2 = pd.DataFrame()
    for j in range(1,6):
        with open(f'results/{model}/{race1}/result_dict{j}.pickle', "rb") as f:
                results = pickle.load(f)
        data_temp = results['data'].drop_duplicates()
        data1 = data1.append(data_temp)
        
        with open(f'results/{model}/{race2}/result_dict{j}.pickle', "rb") as f:
                results = pickle.load(f)
        data_temp = results['data'].drop_duplicates()
        data2 = data2.append(data_temp)        

    return test_type(data1.distance, data2.distance)

def plot_face_heatmap(model, loaded_images):
    e = shap.GradientExplainer(model, data = loaded_images)

    explainer = lime_image.LimeImageExplainer(verbose = True)
    segmenter = SegmentationAlgorithm('slic', n_segments=1000, compactness=0.5, sigma=0)

    explanation = explainer.explain_instance(loaded_images[0].astype('double'), 
                                             classifier_fn = model.predict, 
                                             top_labels=2, hide_color=5, num_samples=10,)# segmentation_fn=segmenter)



    #Select the same class explained on the figures above.
    ind =  explanation.top_labels[0]

    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

    #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.axis('off');
