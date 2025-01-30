import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE
from AQUACULTURE.module import *

def preprocessing_line(processing_line):
    return processing_line.replace("'","").replace(' ', "").replace("[","").replace(']', "")
    
    
def read_feature_list(features): # extracted data by mortality.csv (id and mortality flag)
  return features.iloc[:,0].tolist()


def parse_mortality_flag(mortality_data, id_feature):
  mortality_line = mortality_data.loc[mortality_data['id'] == id_feature]
  return mortality_line.iloc[0].loc['mortality']


def parse_haplotypes_codify(data, id_feature):
  return ((data.loc[data[0] == id_feature]).iloc[0, 1:].astype(str)).values


def count_kmers(sequence, k):
  kmer_count = collections.defaultdict(int)
  for i in range(len(sequence)-(k-1)):
    key = str(sequence[i:i+k])
    kmer_count[key] += 1
  return kmer_count
  
  # Calculate the ratio between max range for frequency and the max frequency of the kmers in fcgr
def ratioFreq(maxFreq, channel_freq):
  return channel_freq / maxFreq
