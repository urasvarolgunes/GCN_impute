# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:28:49 2019

@author: 45467
"""
import pickle
import math
import re
import csv
import concurrent.futures 
import os
from functools import reduce

from operator import add
import pandas as pd
import numpy as np

os.chdir("C:/Users/45467/Desktop/New folder")
ROOT = "./mimic_database/"

## Utilities ##

def map_dict(elem, dictionary):
    if elem in dictionary:
        return dictionary[elem]
    else:
        return np.nan

## Proper Classes ##       
        
class ParseItemID(object):

    ''' This class builds the dictionaries depending on desired features '''
 
    def __init__(self):
        self.dictionary = {}

        self.feature_names = ['RBCs', 'WBCs', 'platelets', 'hemoglobin', 'hemocrit', 
                              'atypical lymphocytes', 'bands', 'basophils', 'eosinophils', 'neutrophils',
                              'lymphocytes', 'monocytes', 'polymorphonuclear leukocytes', 
                              'temperature (F)', 'heart rate', 'respiratory rate', 'systolic', 'diastolic',
                              'pulse oximetry', 
                              'troponin', 'HDL', 'LDL', 'BUN', 'INR', 'PTT', 'PT', 'triglycerides', 'creatinine',
                              'glucose', 'sodium', 'potassium', 'chloride', 'bicarbonate',
                              'blood culture', 'urine culture', 'surface culture', 'sputum' + 
                              ' culture', 'wound culture', 'Inspired O2 Fraction', 'central venous pressure', 
                              'PEEP Set', 'tidal volume', 'anion gap',
                              'daily weight', 'tobacco', 'diabetes', 'history of CV events']

        self.features = ['$^RBC(?! waste)', '$.*wbc(?!.*apache)', '$^platelet(?!.*intake)', 
                         '$^hemoglobin', '$hematocrit(?!.*Apache)', 
                         'Differential-Atyps', 'Differential-Bands', 'Differential-Basos', 'Differential-Eos',
                         'Differential-Neuts', 'Differential-Lymphs', 'Differential-Monos', 'Differential-Polys', 
                         'temperature f', 'heart rate', 'respiratory rate', 'systolic', 'diastolic', 
                         'oxymetry(?! )', 
                         'troponin', 'HDL', 'LDL', '$^bun(?!.*apache)', 'INR', 'PTT',  
                         '$^pt\\b(?!.*splint)(?!.*exp)(?!.*leak)(?!.*family)(?!.*eval)(?!.*insp)(?!.*soft)',
                         'triglyceride', '$.*creatinine(?!.*apache)', 
                         '(?<!boost )glucose(?!.*apache).*',
                       '$^sodium(?!.*apache)(?!.*bicarb)(?!.*phos)(?!.*ace)(?!.*chlo)(?!.*citrate)(?!.*bar)(?!.*PO)',                          '$.*(?<!penicillin G )(?<!urine )potassium(?!.*apache)', 
                         '^chloride', 'bicarbonate', 'blood culture', 'urine culture', 'surface culture',
                         'sputum culture', 'wound culture', 'Inspired O2 Fraction', '$Central Venous Pressure(?! )',
                         'PEEP set', 'tidal volume \(set\)', 'anion gap', 'daily weight', 'tobacco', 'diabetes',
                         'CV - past']                        

        self.patterns = []       
        for feature in self.features:
            if '$' not in feature:
                self.patterns.append('.*{0}.*'.format(feature))
            elif '$' in feature:
                self.patterns.append(feature[1::])

        self.d_items = pd.read_csv(ROOT + 'D_ITEMS.csv', usecols=['ITEMID', 'LABEL'])
        self.d_items.dropna(how='any', axis=0, inplace=True)

        self.script_features_names = ['epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
                                      'asprin', 'ketorolac', 'acetominophen', 
                                      'insulin', 'glucagon', 
                                      'potassium', 'calcium gluconate', 
                                      'fentanyl', 'magensium sulfate', 
                                      'D5W', 'dextrose', 
                                      'ranitidine', 'ondansetron', 'pantoprazole', 'metoclopramide', 
                                      'lisinopril', 'captopril', 'statin',  
                                      'hydralazine', 'diltiazem', 
                                      'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
                                      'amiodarone', 'digoxin(?!.*fab)',
                                      'clopidogrel', 'nitroprusside', 'nitroglycerin',
                                      'vasopressin', 'hydrochlorothiazide', 'furosemide', 
                                      'atropine', 'neostigmine',
                                      'levothyroxine',
                                      'oxycodone', 'hydromorphone', 'fentanyl citrate', 
                                      'tacrolimus', 'prednisone', 
                                      'phenylephrine', 'norepinephrine',
                                      'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
                                      'diazepam', 'clonazepam',
                                      'propofol', 'zolpidem', 'midazolam', 
                                      'albuterol', 'ipratropium', 
                                      'diphenhydramine',  
                                      '0.9% Sodium Chloride',
                                      'phytonadione', 
                                      'metronidazole', 
                                      'cefazolin', 'cefepime', 'vancomycin', 'levofloxacin',
                                      'cipfloxacin', 'fluconazole', 
                                      'meropenem', 'ceftriaxone', 'piperacillin',
                                      'ampicillin-sulbactam', 'nafcillin', 'oxacillin',
                                      'amoxicillin', 'penicillin', 'SMX-TMP']

        self.script_features = ['epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux', 
                                'aspirin', 'keterolac', 'acetaminophen',
                                'insulin', 'glucagon',
                                'potassium', 'calcium gluconate',
                                'fentanyl', 'magnesium sulfate', 
                                'D5W', 'dextrose',   
                                'ranitidine', 'ondansetron', 'pantoprazole', 'metoclopramide', 
                                'lisinopril', 'captopril', 'statin',  
                                'hydralazine', 'diltiazem', 
                                'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
                                'amiodarone', 'digoxin(?!.*fab)',
                                'clopidogrel', 'nitroprusside', 'nitroglycerin',
                                'vasopressin', 'hydrochlorothiazide', 'furosemide', 
                                'atropine', 'neostigmine',
                                'levothyroxine',
                                'oxycodone', 'hydromorphone', 'fentanyl citrate', 
                                'tacrolimus', 'prednisone', 
                                'phenylephrine', 'norepinephrine',
                                'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
                                'diazepam', 'clonazepam',
                                'propofol', 'zolpidem', 'midazolam', 
                                'albuterol', '^ipratropium', 
                                'diphenhydramine(?!.*%)(?!.*cream)(?!.*/)',  
                                '^0.9% sodium chloride(?! )',
                                'phytonadione', 
                                'metronidazole(?!.*%)(?! desensit)', 
                                'cefazolin(?! )', 'cefepime(?! )', 'vancomycin', 'levofloxacin',
                                'cipfloxacin(?!.*ophth)', 'fluconazole(?! desensit)', 
                                'meropenem(?! )', 'ceftriaxone(?! desensit)', 'piperacillin',
                                'ampicillin-sulbactam', 'nafcillin', 'oxacillin', 'amoxicillin',
                                'penicillin(?!.*Desen)', 'sulfamethoxazole']

        self.script_patterns = ['.*' + feature + '.*' for feature in self.script_features]

    def prescriptions_init(self):
        self.prescriptions = pd.read_csv(ROOT + 'PRESCRIPTIONS.csv',
                                         usecols=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'DRUG',
                                                  'STARTDATE', 'ENDDATE'])
        self.prescriptions.dropna(how='any', axis=0, inplace=True)

    def query_prescriptions(self, feature_name):
        pattern = '.*{0}.*'.format(feature_name)
        condition = self.prescriptions['DRUG'].str.contains(pattern, flags=re.IGNORECASE)
        return self.prescriptions['DRUG'].where(condition).dropna().values

    def extractor(self, feature_name, pattern):
        condition = self.d_items['LABEL'].str.contains(pattern, flags=re.IGNORECASE)
        dictionary_value = self.d_items['ITEMID'].where(condition).dropna().values.astype('int')
        self.dictionary[feature_name] = set(dictionary_value)

    def query(self, feature_name):
        pattern = '.*{0}.*'.format(feature_name)
        print(pattern)
        condition = self.d_items['LABEL'].str.contains(pattern, flags=re.IGNORECASE)
        return self.d_items['LABEL'].where(condition).dropna().values

    def query_pattern(self, pattern):
        condition = self.d_items['LABEL'].str.contains(pattern, flags=re.IGNORECASE)
        return self.d_items['LABEL'].where(condition).dropna().values

    def build_dictionary(self): 
        assert len(self.feature_names) == len(self.features)
        for feature, pattern in zip(self.feature_names, self.patterns):
            self.extractor(feature, pattern)

    def reverse_dictionary(self, dictionary):
        self.rev = {}
        for key, value in dictionary.items():
            for elem in value:
                self.rev[elem] = key

class MimicParser(object):
   
    ''' This class structures the MIMIC III and builds features then makes 24 hour windows '''
 
    def __init__(self):
        self.name = 'mimic_assembler'
        self.pid = ParseItemID()
        self.pid.build_dictionary()
        self.features = self.pid.features

    def reduce_total(self, filepath):
       
        ''' This will filter out rows from CHARTEVENTS.csv that are not feauture relevant '''
 
        #CHARTEVENTS = 330712484 

        pid = ParseItemID()
        pid.build_dictionary()
        chunksize = 10000000
        columns = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE',
                   'VALUENUM']

        for i, df_chunk in enumerate(pd.read_csv(filepath, iterator=True, chunksize=chunksize)):
            function = lambda x,y: x.union(y)
            df = df_chunk[df_chunk['ITEMID'].isin(reduce(function, pid.dictionary.values()))]
            df.dropna(inplace=True, axis=0, subset=columns)
            if i == 0:
                df.to_csv(ROOT + './mapped_elements/CHARTEVENTS_reduced.csv', index=False, 
                          columns=columns)
                print(i)
            else:
                df.to_csv(ROOT + './mapped_elements/CHARTEVENTS_reduced.csv', index=False,
                          columns=columns, header=None, mode='a')
                print(i)
            
    def map_files(self, shard_number, filename, low_memory=False):
        
        ''' HADM minimum is 100001 and maximum is 199999. Shards are built off of those. 
            See if can update based on removing rows from previous buckets to accelerate 
            speed (each iteration 10% faster) This may not be necessary of reduce total 
            works well (there are few features)  '''

        buckets = []
        beg = 100001
        end = 199999
        interval = math.ceil((end - beg)/float(shard_number))
       
        for i in np.arange(shard_number):
            buckets.append(set(np.arange(beg+(i*interval),beg+(interval+(interval*i)))))

        if low_memory==False:
            
    
            for i in range(len(buckets)):
                for i,chunk in enumerate(pd.read_csv(filename, iterator=True,
                                                     chunksize=10000000)):
                    print(buckets[i])
                    print(chunk['HADM_ID'].isin(buckets[i]))
                    sliced = chunk[chunk['HADM_ID'].astype('int').isin(buckets[i])] 
                    sliced.to_csv(ROOT + 'mapped_elements/shard_{0}.csv'.format(i), index=False)
                
        else:

            for i in range(len(buckets)):
                with open(filename, 'r') as chartevents:
                    chartevents.seek(0)
                    csvreader = csv.reader(chartevents)
                    with open(ROOT+'mapped_elements/shard_{0}.csv'.format(i), 'w') as shard_writer:
                        csvwriter = csv.writer(shard_writer)
                        for row in csvreader:
                            try:
                                if row[1] == "HADM_ID" or int(row[1]) in buckets[i]:
                                    csvwriter.writerow(row)
                            except ValueError as e:
                                print(row)
                                print(e)
                         
    def create_day_blocks(self, file_name):

        ''' Uses pandas to take shards and build them out '''

        pid = ParseItemID()
        pid.build_dictionary()
        pid.reverse_dictionary(pid.dictionary)
        df = pd.read_csv(file_name)
        df['CHARTDAY'] = df['CHARTTIME'].astype('str').str.split(' ').apply(lambda x: x[0])
        df['HADMID_DAY'] = df['HADM_ID'].astype('str') + '_' + df['CHARTDAY']
        df['FEATURES'] = df['ITEMID'].apply(lambda x: pid.rev[x])
        self.hadm_dict = dict(zip(df['HADMID_DAY'], df['SUBJECT_ID']))
        df2 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', fill_value=np.nan)
        df3 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', aggfunc=np.std, fill_value=0)
        df3.columns = ["{0}_std".format(i) for i in list(df2.columns)]
        df4 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', aggfunc=np.amin, fill_value=np.nan)
        df4.columns = ["{0}_min".format(i) for i in list(df2.columns)]
        df5 = pd.pivot_table(df, index='HADMID_DAY', columns='FEATURES',
                             values='VALUENUM', aggfunc=np.amax, fill_value=np.nan)
        df5.columns = ["{0}_max".format(i) for i in list(df2.columns)]
        df2 = pd.concat([df2, df3, df4, df5], axis=1)
        df2['tobacco'].apply(lambda x: np.around(x))
        del df2['daily weight_std']
        del df2['daily weight_min']
        del df2['daily weight_max']
        del df2['tobacco_std']
        del df2['tobacco_min']
        del df2['tobacco_max']

        rel_columns = list(df2.columns)

        rel_columns = [i for i in rel_columns if '_' not in i]

        for col in rel_columns:
            if len(np.unique(df2[col])[np.isfinite(np.unique(df2[col]))]) <= 2:
                print(col)
                del df2[col + '_std']
                del df2[col + '_min']
                del df2[col + '_max']

        for i in list(df2.columns):
            df2[i][df2[i] > df2[i].quantile(.95)] = df2[i].median()
#            if i != 'troponin':
#                df2[i] = df2[i].where(df2[i] > df2[i].quantile(.875)).fillna(df2[i].median())

        for i in list(df2.columns):
            df2[i].fillna(df2[i].median(), inplace=True)
            
        df2['HADMID_DAY'] = df2.index
        df2['INR'] = df2['INR'] + df2['PT']  
        df2['INR_std'] = df2['INR_std'] + df2['PT_std']  
        df2['INR_min'] = df2['INR_min'] + df2['PT_min']  
        df2['INR_max'] = df2['INR_max'] + df2['PT_max']  
        del df2['PT']
        del df2['PT_std']
        del df2['PT_min']
        del df2['PT_max']
        df2.dropna(thresh=int(0.75*len(df2.columns)), axis=0, inplace=True)
        df2.to_csv(file_name[0:-4] + '_24_hour_blocks.csv', index=False)

    def add_admissions_columns(self, file_name):
        
        ''' Add demographic columns to create_day_blocks '''

        df = pd.read_csv('./mimic_database/ADMISSIONS.csv')
        ethn_dict = dict(zip(df['HADM_ID'], df['ETHNICITY']))
        admittime_dict = dict(zip(df['HADM_ID'], df['ADMITTIME']))
        df_shard = pd.read_csv(file_name)
        df_shard['HADM_ID'] = df_shard['HADMID_DAY'].str.split('_').apply(lambda x: x[0])
        df_shard['HADM_ID'] = df_shard['HADM_ID'].astype('int')
        df_shard['ETHNICITY'] = df_shard['HADM_ID'].apply(lambda x: map_dict(x, ethn_dict))
        black_condition = df_shard['ETHNICITY'].str.contains('.*black.*', flags=re.IGNORECASE)
        df_shard['BLACK'] = 0
        df_shard['BLACK'][black_condition] = 1
        del df_shard['ETHNICITY'] 
        df_shard['ADMITTIME'] = df_shard['HADM_ID'].apply(lambda x: map_dict(x, admittime_dict))
        df_shard.to_csv(file_name[0:-4] + '_plus_admissions.csv', index=False)
                
    def add_patient_columns(self, file_name):
        
        ''' Add demographic columns to create_day_blocks '''

        df = pd.read_csv('./mimic_database/PATIENTS.csv')

        dob_dict = dict(zip(df['SUBJECT_ID'], df['DOB']))
        gender_dict = dict(zip(df['SUBJECT_ID'], df['GENDER']))
        df_shard = pd.read_csv(file_name)
        df_shard['SUBJECT_ID'] = df_shard['HADMID_DAY'].apply(lambda x:
                                                               map_dict(x, self.hadm_dict))
        df_shard['DOB'] = df_shard['SUBJECT_ID'].apply(lambda x: map_dict(x, dob_dict))

        df_shard['YOB'] = df_shard['DOB'].str.split('-').apply(lambda x: x[0]).astype('int')
        df_shard['ADMITYEAR'] = df_shard['ADMITTIME'].str.split('-').apply(lambda x: x[0]).astype('int') 
        
        df_shard['AGE'] = df_shard['ADMITYEAR'].subtract(df_shard['YOB']) 
        df_shard['GENDER'] = df_shard['SUBJECT_ID'].apply(lambda x: map_dict(x, gender_dict))
        gender_dummied = pd.get_dummies(df_shard['GENDER'], drop_first=True)
        gender_dummied.rename(columns={'M': 'Male', 'F': 'Female'})
        COLUMNS = list(df_shard.columns)
        COLUMNS.remove('GENDER')
        df_shard = pd.concat([df_shard[COLUMNS], gender_dummied], axis=1)
        df_shard.to_csv(file_name[0:-4] + '_plus_patients.csv', index=False)

    def clean_prescriptions(self, file_name):
        
        ''' Add prescriptions '''

        pid = ParseItemID()
        pid.prescriptions_init()        
        pid.prescriptions.drop_duplicates(inplace=True)
        pid.prescriptions['DRUG_FEATURE'] = np.nan

        df_file = pd.read_csv(file_name)
        hadm_id_array = pd.unique(df_file['HADM_ID'])

        for feature, pattern in zip(pid.script_features_names, pid.script_patterns):
           condition = pid.prescriptions['DRUG'].str.contains(pattern, flags=re.IGNORECASE)
           pid.prescriptions['DRUG_FEATURE'][condition] = feature

        pid.prescriptions.dropna(how='any', axis=0, inplace=True, subset=['DRUG_FEATURE']) 

        pid.prescriptions.to_csv('./mimic_database/PRESCRIPTIONS_reduced.csv', index=False)
        
    def add_prescriptions(self, file_name):
       
        df_file = pd.read_csv(file_name)

        with open('./mimic_database/PRESCRIPTIONS_reduced.csv', 'r') as f:
            csvreader = csv.reader(f)
            with open('./mimic_database/PRESCRIPTIONS_reduced_byday.csv', 'w') as g:
                csvwriter  = csv.writer(g)
                first_line = csvreader.__next__()
                print(first_line[0:3] + ['CHARTDAY'] + [first_line[6]])
                csvwriter.writerow(first_line[0:3] + ['CHARTDAY'] + [first_line[6]])
                for row in csvreader:
                    for i in pd.date_range(row[3], row[4]).strftime('%Y-%m-%d'):
                        csvwriter.writerow(row[0:3] + [i] + [row[6]])

        df = pd.read_csv('./mimic_database/PRESCRIPTIONS_reduced_byday.csv')
        df['CHARTDAY'] = df['CHARTDAY'].str.split(' ').apply(lambda x: x[0])
        df['HADMID_DAY'] = df['HADM_ID'].astype('str') + '_' + df['CHARTDAY']
        df['VALUE'] = 1        

        cols = ['HADMID_DAY', 'DRUG_FEATURE', 'VALUE']
        df = df[cols] 
 
        df_pivot = pd.pivot_table(df, index='HADMID_DAY', columns='DRUG_FEATURE', values='VALUE', fill_value=0, aggfunc=np.amax)
        df_pivot.reset_index(inplace=True)

        df_merged = pd.merge(df_file, df_pivot, on='HADMID_DAY', how='outer')
 
        del df_merged['HADM_ID']
        df_merged['HADM_ID'] = df_merged['HADMID_DAY'].str.split('_').apply(lambda x: x[0])
        df_merged.fillna(0, inplace=True)

        df_merged['dextrose'] = df_merged['dextrose'] + df_merged['D5W']
        del df_merged['D5W']

        df_merged.to_csv(file_name[0:-4] + '_plus_scripts.csv', index=False)

    def add_icd_infect(self, file_name):

        df_icd = pd.read_csv('./mimic_database/PROCEDURES_ICD.csv')
        df_micro = pd.read_csv('./mimic_database/MICROBIOLOGYEVENTS.csv')
        self.suspect_hadmid = set(pd.unique(df_micro['HADM_ID']).tolist())
        df_icd_ckd = df_icd[df_icd['ICD9_CODE'] == 585]
        
        self.ckd = set(df_icd_ckd['HADM_ID'].values.tolist())
        
        df = pd.read_csv(file_name)
        df['CKD'] = df['HADM_ID'].apply(lambda x: 1 if x in self.ckd else 0)
        df['Infection'] = df['HADM_ID'].apply(lambda x: 1 if x in self.suspect_hadmid else 0)
        df.to_csv(file_name[0:-4] + '_plus_icds.csv', index=False)

    def add_notes(self, file_name):
        df = pd.read_csv('./mimic_database/NOTEEVENTS.csv')
        df_rad_notes = df[['TEXT', 'HADM_ID']][df['CATEGORY'] == 'Radiology']
        CTA_bool_array = df_rad_notes['TEXT'].str.contains('CTA', flags=re.IGNORECASE)
        CT_angiogram_bool_array = df_rad_notes['TEXT'].str.contains('CT angiogram', flags=re.IGNORECASE)
        chest_angiogram_bool_array = df_rad_notes['TEXT'].str.contains('chest angiogram', flags=re.IGNORECASE)
        cta_hadm_ids = np.unique(df_rad_notes['HADM_ID'][CTA_bool_array].dropna())
        CT_angiogram_hadm_ids = np.unique(df_rad_notes['HADM_ID'][CT_angiogram_bool_array].dropna())
        chest_angiogram_hadm_ids = np.unique(df_rad_notes['HADM_ID'][chest_angiogram_bool_array].dropna())
        hadm_id_set = set(cta_hadm_ids.tolist())
        hadm_id_set.update(CT_angiogram_hadm_ids)
        print(len(hadm_id_set))
        hadm_id_set.update(chest_angiogram_hadm_ids)
        print(len(hadm_id_set))
        
        df2 = pd.read_csv(file_name)
        df2['ct_angio'] = df2['HADM_ID'].apply(lambda x: 1 if x in hadm_id_set else 0)
        df2.to_csv(file_name[0:-4] + '_plus_notes.csv', index=False)

if __name__ == '__main__':

    pid = ParseItemID()
    pid.build_dictionary()
    FOLDER = 'mapped_elements/'
    FILE_STR = 'CHARTEVENTS_reduced'
    mp = MimicParser()

    mp.reduce_total(ROOT + 'CHARTEVENTS.csv')
    mp.create_day_blocks(ROOT+ FOLDER + FILE_STR + '.csv')
    mp.add_admissions_columns(ROOT + FOLDER + FILE_STR + '_24_hour_blocks.csv')
    mp.add_patient_columns(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions.csv')
    mp.clean_prescriptions(ROOT + FOLDER + FILE_STR + 
                         '_24_hour_blocks_plus_admissions_plus_patients.csv')
    mp.add_prescriptions(ROOT + FOLDER + FILE_STR + 
                         '_24_hour_blocks_plus_admissions_plus_patients.csv')
    mp.add_icd_infect(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions_plus_patients_plus_scripts.csv') 
    mp.add_notes(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds.csv')   
    
    
    
''' Meant to deal with padding for an RNN when keras.preprocessing.pad_sequences fails '''

import numpy as np
import pandas as pd

class PadSequences(object):
    
    def __init__(self):
        self.name = 'padder'

    def pad(self, df, lb, time_steps, pad_value=-100):

        ''' Takes a file path for the dataframe to operate on. lb is a lower bound to discard 
            ub is an upper bound to truncate on. All entries are padded to their ubber bound '''

        #df = pd.read_csv(path):
        self.uniques = pd.unique(df['HADM_ID'])
        df = df.groupby('HADM_ID').filter(lambda group: len(group) > lb).reset_index(drop=True)
        df = df.groupby('HADM_ID').apply(lambda group: group[0:time_steps]).reset_index(drop=True)
        df = df.groupby('HADM_ID').apply(lambda group: pd.concat([group, pd.DataFrame(pad_value*np.ones((time_steps-len(group), len(df.columns))), columns=df.columns)], axis=0)).reset_index(drop=True)
         
        return df

    def ZScoreNormalize(self, matrix):

        ''' Performs Z Score Normalization for 3rd order tensors 
            matrix should be (batchsize, time_steps, features) 
            Padded time steps should be masked with np.nan '''

        x_matrix = matrix[:,:,0:-1]
        y_matrix = matrix[:,:,-1]
        print(y_matrix.shape)
        y_matrix = y_matrix.reshape(y_matrix.shape[0],y_matrix.shape[1],1)
        means = np.nanmean(x_matrix, axis=(0,1))
        stds = np.nanstd(x_matrix, axis=(0,1))
        print(x_matrix.shape)
        print(means.shape)
        print(stds.shape)
        x_matrix = x_matrix-means
        print(x_matrix.shape)
        x_matrix = x_matrix / stds
        print(x_matrix.shape)
        print(y_matrix.shape)
        matrix = np.concatenate([x_matrix, y_matrix], axis=2)
        
        return matrix
    
    def MinMaxScaler(self, matrix, pad_value=-100):

        ''' Performs a NaN/pad-value insensiive MinMaxScaling 
            When column maxs are zero, it ignores these columns for division '''

        bool_matrix = (matrix == pad_value)
        matrix[bool_matrix] = np.nan
        mins = np.nanmin(matrix, axis=0)
        maxs = np.nanmax(matrix, axis=0)
        matrix = np.divide(np.subtract(matrix, mins), np.subtract(maxs,mins), where=(np.nanmax(matrix,axis=0) != 0))
        matrix[bool_matrix] = pad_value

        return matrix
    
    
    
''' Recurrent Neural Network in Keras for use on the MIMIC-III '''

import gc
from time import time
import os
import math
import pickle

import numpy as np
import pandas as pd
from pad_sequences import PadSequences
#from processing_utilities import PandasUtilities
from attention_function import attention_3d_block as Attention 

from keras import backend as K
from keras.models import Model, Input, load_model #model_from_json
from keras.layers import Masking, Flatten, Embedding, Dense, LSTM, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import optimizers

#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = "./mimic_database/mapped_elements/"
FILE = "CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"

######################################
## MAIN ###
######################################

def get_synth_sequence(n_timesteps=14):

  """
  Returns a single synthetic data sequence of dim (bs,ts,feats)
  Args:
  ----
    n_timesteps: int, number of timesteps to build model for
  Returns:
  -------
    X: npa, numpy array of features of shape (1,n_timesteps,2)
    y: npa, numpy array of labels of shape (1,n_timesteps,1) 
  """

  X = np.array([[np.random.rand() for _ in range(n_timesteps)],[np.random.rand() for _ in range(n_timesteps)]])
  X = X.reshape(1, n_timesteps, 2)
  y = np.array([0 if x.sum() < 0.5 else 1 for x in X[0]])
  y = y.reshape(1, n_timesteps, 1)
  return X, y

def wbc_crit(x):
  if (x > 12 or x < 4) and x != 0:
    return 1
  else:
    return 0

def temp_crit(x):
  if (x > 100.4 or x < 96.8) and x != 0:
    return 1
  else:
    return 0

def return_data(synth_data=False, balancer=True, target='MI', 
                return_cols=False, tt_split=0.7, val_percentage=0.8,
                cross_val=False, mask=False, dataframe=False,
                time_steps=14, split=True, pad=True):

  """
  Returns synthetic or real data depending on parameter
  Args:
  -----
      synth_data : synthetic data is False by default
      balance : whether or not to balance positive and negative time windows 
      target : desired target, supports MI, SEPSIS, VANCOMYCIN or a known lab, medication
      return_cols : return columns used for this RNN
      tt_split : fraction of dataset to use fro training, remaining is used for test
      cross_val : parameter that returns entire matrix unsplit and unbalanced for cross val purposes
      mask : 24 hour mask, default is False
      dataframe : returns dataframe rather than numpy ndarray
      time_steps : 14 by default, required for padding
      split : creates test train splits
      pad : by default is True, will pad to the time_step value
  Returns:
  -------
      Training and validation splits as well as the number of columns for use in RNN  
  """

  if synth_data:
    no_feature_cols = 2
    X_train = []
    y_train = []

    for i in range(10000):
      X, y = get_synth_sequence(n_timesteps=14)
      X_train.append(X)
      y_train.append(y)
    X_TRAIN = np.vstack(X_train)
    Y_TRAIN = np.vstack(y_train)

  else:
    df = pd.read_csv(ROOT + FILE)

    if target == 'MI':
      df[target] = ((df['troponin'] > 0.4) & (df['CKD'] == 0)).apply(lambda x: int(x))

    elif target == 'SEPSIS':
      df['hr_sepsis'] = df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
      df['respiratory rate_sepsis'] = df['respiratory rate'].apply(lambda x: 1 if x>20 else 0)
      df['wbc_sepsis'] = df['WBCs'].apply(wbc_crit) 
      df['temperature f_sepsis'] = df['temperature (F)'].apply(temp_crit) 
      df['sepsis_points'] = (df['hr_sepsis'] + df['respiratory rate_sepsis'] 
                          + df['wbc_sepsis'] + df['temperature f_sepsis'])
      df[target] = ((df['sepsis_points'] >= 2) & (df['Infection'] == 1)).apply(lambda x: int(x))
      del df['hr_sepsis']
      del df['respiratory rate_sepsis']
      del df['wbc_sepsis']
      del df['temperature f_sepsis']
      del df['sepsis_points']
      del df['Infection']

    elif target == 'PE':
      df['blood_thinner'] = (df['heparin']  + df['enoxaparin'] + df['fondaparinux']).apply(lambda x: 1 if x >= 1 else 0)
      df[target] = (df['blood_thinner'] & df['ct_angio']) 
      del df['blood_thinner']


    elif target == 'VANCOMYCIN':
      df['VANCOMYCIN'] = df['vancomycin'].apply(lambda x: 1 if x > 0 else 0)   
      del df['vancomycin']
 
    df = df.select_dtypes(exclude=['object'])

    if pad:
      pad_value=0
      df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
      print('There are {0} rows in the df after padding'.format(len(df)))

    COLUMNS = list(df.columns)

    if target == 'MI':
      toss = ['ct_angio', 'troponin', 'troponin_std', 'troponin_min', 'troponin_max', 'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'SEPSIS':
      toss = ['ct_angio', 'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'PE':
      toss = ['ct_angio', 'heparin', 'heparin_std', 'heparin_min',
              'heparin_max', 'enoxaparin', 'enoxaparin_std',
              'enoxaparin_min', 'enoxaparin_max', 'fondaparinux',
              'fondaparinux_std', 'fondaparinux_min', 'fondaparinux_max',
              'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'VANCOMYCIN':
      toss = ['ct_angio', 'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]

    COLUMNS.remove(target)

    if 'HADM_ID' in COLUMNS:
      COLUMNS.remove('HADM_ID')
    if 'SUBJECT_ID' in COLUMNS:
      COLUMNS.remove('SUBJECT_ID')
    if 'YOB' in COLUMNS:
      COLUMNS.remove('YOB')
    if 'ADMITYEAR' in COLUMNS:
      COLUMNS.remove('ADMITYEAR')

    if return_cols:
      return COLUMNS

    if dataframe:
      return (df[COLUMNS+[target,"HADM_ID"]]) 


    MATRIX = df[COLUMNS+[target]].values
    MATRIX = MATRIX.reshape(int(MATRIX.shape[0]/time_steps),time_steps,MATRIX.shape[1])

    ## note we are creating a second order bool matirx
    bool_matrix = (~MATRIX.any(axis=2))
    MATRIX[bool_matrix] = np.nan
    MATRIX = PadSequences().ZScoreNormalize(MATRIX)
    ## restore 3D shape to boolmatrix for consistency
    bool_matrix = np.isnan(MATRIX)
    MATRIX[bool_matrix] = pad_value 
   
    permutation = np.random.permutation(MATRIX.shape[0])
    MATRIX = MATRIX[permutation]
    bool_matrix = bool_matrix[permutation]

    X_MATRIX = MATRIX[:,:,0:-1]
    Y_MATRIX = MATRIX[:,:,-1]
    
    x_bool_matrix = bool_matrix[:,:,0:-1]
    y_bool_matrix = bool_matrix[:,:,-1]

    X_TRAIN = X_MATRIX[0:int(tt_split*X_MATRIX.shape[0]),:,:]
    Y_TRAIN = Y_MATRIX[0:int(tt_split*Y_MATRIX.shape[0]),:]
    Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    X_VAL = X_MATRIX[int(tt_split*X_MATRIX.shape[0]):int(val_percentage*X_MATRIX.shape[0])]
    Y_VAL = Y_MATRIX[int(tt_split*Y_MATRIX.shape[0]):int(val_percentage*Y_MATRIX.shape[0])]
    Y_VAL = Y_VAL.reshape(Y_VAL.shape[0], Y_VAL.shape[1], 1)

    x_val_boolmat = x_bool_matrix[int(tt_split*x_bool_matrix.shape[0]):int(val_percentage*x_bool_matrix.shape[0])]
    y_val_boolmat = y_bool_matrix[int(tt_split*y_bool_matrix.shape[0]):int(val_percentage*y_bool_matrix.shape[0])]
    y_val_boolmat = y_val_boolmat.reshape(y_val_boolmat.shape[0],y_val_boolmat.shape[1],1)

    X_TEST = X_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
    Y_TEST = Y_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
    Y_TEST = Y_TEST.reshape(Y_TEST.shape[0], Y_TEST.shape[1], 1)

    x_test_boolmat = x_bool_matrix[int(val_percentage*x_bool_matrix.shape[0])::]
    y_test_boolmat = y_bool_matrix[int(val_percentage*y_bool_matrix.shape[0])::]
    y_test_boolmat = y_test_boolmat.reshape(y_test_boolmat.shape[0],y_test_boolmat.shape[1],1)

    X_TEST[x_test_boolmat] = pad_value
    Y_TEST[y_test_boolmat] = pad_value

    if balancer:
      TRAIN = np.concatenate([X_TRAIN, Y_TRAIN], axis=2)
      print(np.where((TRAIN[:,:,-1] == 1).any(axis=1))[0])
      pos_ind = np.unique(np.where((TRAIN[:,:,-1] == 1).any(axis=1))[0])
      print(pos_ind)
      np.random.shuffle(pos_ind)
      neg_ind = np.unique(np.where(~(TRAIN[:,:,-1] == 1).any(axis=1))[0])
      print(neg_ind)
      np.random.shuffle(neg_ind)
      length = min(pos_ind.shape[0], neg_ind.shape[0])
      total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
      np.random.shuffle(total_ind)
      ind = total_ind
      if target == 'MI':
        ind = pos_ind
      else:
        ind = total_ind
      X_TRAIN = TRAIN[ind,:,0:-1]
      Y_TRAIN = TRAIN[ind,:,-1]
      Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

  no_feature_cols = X_TRAIN.shape[2]

  if mask:
    print('MASK ACTIVATED')
    X_TRAIN = np.concatenate([np.zeros((X_TRAIN.shape[0], 1, X_TRAIN.shape[2])), X_TRAIN[:,1::,::]], axis=1)
    X_VAL = np.concatenate([np.zeros((X_VAL.shape[0], 1, X_VAL.shape[2])), X_VAL[:,1::,::]], axis=1)

  if cross_val:
    return (MATRIX, no_feature_cols)
    
  if split == True:
    return (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
            X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat,
            x_val_boolmat, y_val_boolmat)

  elif split == False:
    return (np.concatenate((X_TRAIN,X_VAL), axis=0),
            np.concatenate((Y_TRAIN,Y_VAL), axis=0), no_feature_cols) 

def build_model(no_feature_cols=None, time_steps=7, output_summary=False):

  """
  Assembles RNN with input from return_data function
  Args:
  ----
  no_feature_cols : The number of features being used AKA matrix rank
  time_steps : The number of days in a time block
  output_summary : Defaults to False on returning model summary
  Returns:
  ------- 
  Keras model object
  """
  print("time_steps:{0}|no_feature_cols:{1}".format(time_steps,no_feature_cols)) 
  input_layer = Input(shape=(time_steps, no_feature_cols)) 
  x = Attention(input_layer, time_steps)
  x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(x) 
  x = LSTM(256, return_sequences=True)(x)
  preds = TimeDistributed(Dense(1, activation="sigmoid"))(x)
  model = Model(inputs=input_layer, outputs=preds)
  
  RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
  model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])
  
  if output_summary:
    model.summary()
  return model

def train(model_name="kaji_mach_0", synth_data=False, target='MI',
          balancer=True, predict=False, return_model=False,
          n_percentage=1.0, time_steps=14, epochs=10):

  """
  Use Keras model.fit using parameter inputs
  Args:
  ----
  model_name : Parameter used for naming the checkpoint_dir
  synth_data : Default to False. Allows you to use synthetic or real data.
  Return:
  -------
  Nonetype. Fits model only. 
  """

  f = open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'rb')
  X_TRAIN = pickle.load(f)
  f.close()

  f = open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'rb')
  Y_TRAIN = pickle.load(f)
  f.close()
  
  f = open('./pickled_objects/X_VAL_{0}.txt'.format(target), 'rb')
  X_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'rb')
  Y_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'rb')
  X_BOOLMAT_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'rb')
  Y_BOOLMAT_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'rb')
  no_feature_cols = pickle.load(f)
  f.close()

  X_TRAIN = X_TRAIN[0:int(n_percentage*X_TRAIN.shape[0])]
  Y_TRAIN = Y_TRAIN[0:int(n_percentage*Y_TRAIN.shape[0])]

  #build model
  model = build_model(no_feature_cols=no_feature_cols, output_summary=True, 
                      time_steps=time_steps)

  #init callbacks
  tb_callback = TensorBoard(log_dir='./logs/{0}_{1}.log'.format(model_name, time),
    histogram_freq=0,
    write_grads=False,
    write_images=True,
    write_graph=True) 

  #Make checkpoint dir and init checkpointer
  checkpoint_dir = "./saved_models/{0}".format(model_name)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  checkpointer = ModelCheckpoint(
    filepath=checkpoint_dir+"/model.{epoch:02d}-{val_loss:.2f}.hdf5",
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)

  #fit
  model.fit(
    x=X_TRAIN,
    y=Y_TRAIN,
    batch_size=16,
    epochs=epochs,
    callbacks=[tb_callback], #, checkpointer],
    validation_data=(X_VAL, Y_VAL),
    shuffle=True)

  model.save('./saved_models/{0}.h5'.format(model_name))

  if predict:
    print('TARGET: {0}'.format(target))
    Y_PRED = model.predict(X_VAL)
    Y_PRED = Y_PRED[~Y_BOOLMAT_VAL]
    np.unique(Y_PRED)
    Y_VAL = Y_VAL[~Y_BOOLMAT_VAL]
    Y_PRED_TRAIN = model.predict(X_TRAIN)
    print('Confusion Matrix Validation')
    print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
    print('Validation Accuracy')
    print(accuracy_score(Y_VAL, np.around(Y_PRED)))
    print('ROC AUC SCORE VAL')
    print(roc_auc_score(Y_VAL, Y_PRED))
    print('CLASSIFICATION REPORT VAL')
    print(classification_report(Y_VAL, np.around(Y_PRED)))

  if return_model:
    return model

def return_loaded_model(model_name="kaji_mach_0"):

  loaded_model = load_model("./saved_models/{0}.h5".format(model_name))

  return loaded_model

def pickle_objects(target='MI', time_steps=14):

  (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
   X_TEST, Y_TEST, x_boolmat_test, y_boolmat_test,
   x_boolmat_val, y_boolmat_val) = return_data(balancer=True, target=target,
                                                            pad=True,
                                                            split=True, 
                                                      time_steps=time_steps)  

  features = return_data(return_cols=True, synth_data=False,
                         target=target, pad=True, split=True,
                         time_steps=time_steps)

  f = open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'wb')
  pickle.dump(X_TRAIN, f)
  f.close()

  f = open('./pickled_objects/X_VAL_{0}.txt'.format(target), 'wb')
  pickle.dump(X_VAL, f)
  f.close()

  f = open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'wb')
  pickle.dump(Y_TRAIN, f)
  f.close()

  f = open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'wb')
  pickle.dump(Y_VAL, f)
  f.close()

  f = open('./pickled_objects/X_TEST_{0}.txt'.format(target), 'wb')
  pickle.dump(X_TEST, f)
  f.close()

  f = open('./pickled_objects/Y_TEST_{0}.txt'.format(target), 'wb')
  pickle.dump(Y_TEST, f)
  f.close()

  f = open('./pickled_objects/x_boolmat_test_{0}.txt'.format(target), 'wb')
  pickle.dump(x_boolmat_test, f)
  f.close()

  f = open('./pickled_objects/y_boolmat_test_{0}.txt'.format(target), 'wb')
  pickle.dump(y_boolmat_test, f)
  f.close()

  f = open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'wb')
  pickle.dump(x_boolmat_val, f)
  f.close()

  f = open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'wb')
  pickle.dump(y_boolmat_val, f)
  f.close()

  f = open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'wb')
  pickle.dump(no_feature_cols, f)
  f.close()

  f = open('./pickled_objects/features_{0}.txt'.format(target), 'wb')
  pickle.dump(features, f)
  f.close()


if __name__ == "__main__":

    pickle_objects(target='MI', time_steps=14)#
    K.clear_session()
    pickle_objects(target='SEPSIS', time_steps=14)
    K.clear_session()
    pickle_objects(target='VANCOMYCIN', time_steps=14)

## BIG THREE ##

    K.clear_session()
    train(model_name='kaji_mach_final_no_mask_MI_pad14', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14)

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14', epochs=14,
          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14) 

    K.clear_session()

    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14', epochs=17,
          synth_data=False, predict=True, target='SEPSIS', time_steps=14) 


## REDUCE SAMPLE SIZES ##

## MI ##

    train(model_name='kaji_mach_final_no_mask_MI_pad14_80_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.80)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_60_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.60)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_40_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.40)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_20_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.20)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_10_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.10)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_5_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.05)
  
    K.clear_session()
  
# SEPSIS ##
 
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_80_percent',
          epochs=14,synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.80) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_60_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.60) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_40_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.40) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_20_percent', epochs=14,
          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14,
          n_percentage=0.20) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_10_percent',
          epochs=13, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.10)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_5_percent',
          epochs=13, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.05)
 
# VANCOMYCIN ##
 
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_80_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.80) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_60_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.60) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_40_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.40) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_20_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.20) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_10_percent',
          epochs=13, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.10)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_5_percent',
          epochs=13, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.05)
