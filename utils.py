from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import random
import re
import scipy.stats as st
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import tensorflow as tf

#import cv2
from deepface import DeepFace



def create_base_race_df(race, size = 1000):
    """Create dataframe for a given race.

    race (str): Ethnicity for which the dataframe needs to be created
    size (int): Count of instances for a dataframe

    """
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

def create_same_person_dataset(base_data):
    """Create dataframe for same person.

    base_data (pd.DataFrame): Dataframe including column with image paths

    """
    
    # create random same person df
    same_person_df = base_data.sample(int(len(base_data)/2))
    same_person_df['compare_person'] =same_person_df.original_person

    # get random image from same person
    same_person_df['compare_image'] = same_person_df.original_person.apply(lambda x: random.choice(os.listdir(x)))

    # create total image path
    same_person_df['compare_image'] = same_person_df['compare_person']+'/'+same_person_df['compare_image']
    same_person_df['is_same_person'] =True
    
    return same_person_df

def create_different_person_dataset(base_data):

    """Create dataframe for comparing different persons.

    base_data (pd.DataFrame): Dataframe including column with image paths

    """
    
    # randomly assing images to entire dataset
    base_data['compare_image'] = shuffle(base_data.original_image)
    
    # Define regex "including until 4th slash"
    p = re.compile('.*/.*/.*/.*/')
    # cut total image path to only return person
    base_data['compare_person'] = base_data['compare_image'].apply(lambda x:
                                                                p.findall(x)[0][:-1])

    base_data['is_same_person'] = False
    # Only filter rows with different persons
    different_person_df = base_data[
        (base_data.compare_person!=base_data.original_person)].drop_duplicates().sample(int(len(df)/2))
    
    return different_person_df

def create_evaluation_dataset(base_data):
    """Create final dataset including positive and negative image pairs.

    base_data (pd.DataFrame): Dataframe including column with image paths

    """

    # create random same person df
    same_person_df = create_same_person_dataset(base_data)
    
    # create random different person df
    different_person_df = create_different_person_dataset(base_data)
    
    # Create target variable
    evaluation = same_person_df.append(different_person_df)
    evaluation.is_same_person = evaluation.is_same_person.fillna(False)
    
    print('Target distribution is:')
    print(evaluation.is_same_person.value_counts(normalize=True))
    
    return evaluation

# PREDICTION FUNCTIONS

def normalize_distances(base_data, distance_column = 'distance', threshold = .5):
    """Normalizing distance metric to scale values to 0 and 1.

    base_data (pd.DataFrame): Dataframe including column with image paths
    distance_column (str): Name of the column which needs to be normalized
    threshold (float): Normalized score threshold to be used for classifying into positive and negative

    """
    
    base_data['distance_normalised'] = MinMaxScaler().fit_transform(np.array(base_data[distance_column]).reshape(-1, 1))
    base_data.loc[base_data.distance_normalised>threshold,"verified_normalised"] = False
    base_data.loc[base_data.distance_normalised<threshold,"verified_normalised"] = True

    return base_data

def verify_faces(base_data, 
                model, 
                performance_metrics= [roc_auc_score, accuracy_score], 
                face_count= 100, 
                distance_metric='cosine', 
                preprocessing  = None):
    """Calculate distance between image pairs.

    base_data (pd.DataFrame): Dataframe including positive and negative image pairs 
    model (loaded model): Loaded model
    performance_metrics (list): List of functions of evaluation metrics
    face_count (str): Batch size for face verfication
    distance_metric (str): Distance function to be used for similarity calculation
    preprocessing (function): Preprocessing method to be applied

    """

    df['distance'] = np.nan
    df['verified'] = np.nan

    df=df.sample(frac=1)[:face_count]

    for i in range(0,len(df)):
        if preprocessing is None:
            verification_result = DeepFace.verify(
                img1_path = df.iloc[i].original_image,
                img2_path = df.iloc[i].compare_image,
                distance_metric = distance_metric,
                model = model,
                enforce_detection=False)
        else:
            img1 = preprocessing(df.iloc[i].original_image)
            img2 = preprocessing(df.iloc[i].compare_image)
            
            verification_result = DeepFace.verify(
                img1_path = img1,
                img2_path = img2,
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
def calculate_confidence_interval(model, race, metric=roc_auc_score):
    """Calculate 95% confidence interval for a given evaluation metric.

    model (str): Name of the model 
    race (str): Name of Race
    performance_metrics (list): List of functions of evaluation metrics
    metric (function): Evaluation function to be applied

    """

    directory = f"results/{model}/{race}"
    y_pred = []
    y_true = []
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            with open(os.path.join(directory,file), "rb") as f:
                results = pickle.load(f)
            y_pred.append(results["data"].distance)
            y_true.append(results["data"].is_same_person)
    y_pred = np.array(y_pred).ravel()
    y_true = np.array(y_true).ravel()



    n_bootstraps = 300
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), 1000)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if metric == accuracy_score:
            score = metric(y_true[indices], np.round(1-y_pred[indices]))
        else:
            score = metric(y_true[indices], 1-y_pred[indices])
        bootstrapped_scores.append(score)

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower, confidence_upper = st.t.interval(0.95, len(bootstrapped_scores)-1, loc=np.mean(bootstrapped_scores), scale=st.sem(bootstrapped_scores))
    
    return np.mean(bootstrapped_scores), confidence_lower, confidence_upper
    


def calculate_performance_per_threshold(model, race):
    """Calculate the performance of evaluation metrics for different thresholds.

    model (str): Name of the model 
    race (str): Name of Race

    """

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
    """Group and count distances.

    distances (pd.DataFrame): Dataframe containing the calculated distances in column named 'distances'
    groupby (str): Column on which the dataframe will be grouped by

    """
    
    distances['distance'] = MinMaxScaler().fit_transform(np.array(distances.distance).reshape(-1, 1))
    distances = round(distances,1)

    grouped_distances = distances.groupby(groupby).original_image.count()

    grouped_distances = grouped_distances.reset_index()
    return grouped_distances

def count_distances(model, race, groupby = ['distance']):
    """Count number of distances.

    model (str): Name of the model 
    race (str): Name of Race
    groupby (str): Column on which the dataframe will be grouped by

    """

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
    """Calculate differences between distributions.

    model (str): Name of the model 
    race (str): Name of Race

    """
    directory = f"results/{model}/{race}"
    distances = pd.DataFrame()
    
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            with open(os.path.join(directory,file), "rb") as f:
                results = pickle.load(f)
            data = results['data']
            distances = distances.append(pd.DataFrame(data))
    

    return distances.groupby("is_same_person").distance.mean().loc[True] - distances.groupby("is_same_person").distance.mean().loc[False]

def plot_performance_per_threshold(model):
    """Plot performance for different thresholds.

    model (str): Name of the model 

    """

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
    plt.show()

def get_metric_comparision(model, metric):
    """Create a summary for a given model and metric for all ethnicities.

    model (str): Name of the model
    metric (function): Evaluation function to be applied

    """
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
    """Plot score distribution.

    model (str): Name of the model 
    groupby (list): Column as list which will be grouped
    hue (str): Column on which the hue will be applied

    """

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
    """Plot score distribution.

    model (str): Name of the model 
    race1 (str): Name of first race
    race2 (str): Name of second race
    test_type (function): Function for statistical significance test

    """
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
    """Plot score distribution.

    model (str): Name of the model 
    loaded_images (function): Loaded images

    """

    explainer = lime_image.LimeImageExplainer(verbose = True)
    segmenter = SegmentationAlgorithm('slic', n_segments=1000, compactness=1, sigma=-1)

    explanation = explainer.explain_instance(loaded_images.astype('double'), 
                                             classifier_fn = model.predict, 
                                             top_labels=2, hide_color=4, num_samples=30,)# segmentation_fn=segmenter)



    #Select the same class explained on the figures above.
    ind =  explanation.top_labels[0]

    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

    #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.axis('off')
