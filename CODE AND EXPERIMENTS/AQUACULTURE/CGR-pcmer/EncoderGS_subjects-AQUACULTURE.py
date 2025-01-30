
## MODULE
import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE

from AQUACULTURE.module import * 
from AQUACULTURE.cgr_haplotype import CGR_HAPLOTYPE_4POINTS
#from cgr_haplotype_3_points import CGR_HAPLOTYPE
from AQUACULTURE.fcgr_GS import FCGR_GS
from AQUACULTURE.functions_FCGR_AQUACULTURE import parse_mortality_flag, read_feature_list, parse_haplotypes_codify

if __name__ == '__main__':
    
    alive1_C1 = 'PL10-A03'
    alive2_C1 = 'PL15-G01'
    dead1_C1 = 'PL09-H11'
    dead2_C1 = 'PL05-E05'

    alive1_C0 = 'PL08n-B04'
    alive2_C0 = 'PL11-B06'
    dead1_C0 = 'PL04-A06'
    dead2_C0 = 'PL05-D12'

    peak = "_specific80"

    MAIN_FOLDER = 'CODE AND EXPERIMENTS'
    out_directory = MAIN_FOLDER + '/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/include-chr3'
    path_datasets = MAIN_FOLDER + '/DATASET/PNRR'

    features = pd.read_csv(path_datasets + '/mortality.csv', skiprows=1, header=None)
    print('Open features')
    mortality_data = pd.read_csv(path_datasets + '/mortality.csv')
    print('Open mortality')
    key = dead2_C0
    data = pd.read_csv(path_datasets + '/' + key + peak + '.csv', header=None, low_memory=False)
    print('Ok extracted datasets')
    

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    selected_features = read_feature_list(features)
    #print('Extracted features in mortality')
        
    for id_feature in selected_features:
        #print('id', id_feature)
        if id_feature != key: print(id_feature); continue
        title = id_feature

        w, h = 1024, 1024
       
        flag_mortality = parse_mortality_flag(mortality_data, id_feature)
        #print('Extreacted mortality flag', type(flag_mortality), flag_mortality)
        if flag_mortality == 0: dir_mortality = "ALIVE" # vivo
        elif flag_mortality == 1: dir_mortality = "DEAD" # morto
        #print('dir', dir_mortality)

        codify_haplotype = parse_haplotypes_codify(data, id_feature)
            
        print('Extracted haplotype codify', codify_haplotype, type(codify_haplotype), len(codify_haplotype))
        
        kmers = [2]

        for k in kmers:
            directory_png = out_directory + '/CGR/' + 'GS 4 points/' + dir_mortality
            #directory_png = out_directory + '/CGR/' + 'GS 3 points/' + dir_mortality
            #directory_png = out_directory + '/' + 'FCGR GS (k='+ str(k) + ')/' + dir_mortality
            if not os.path.exists(directory_png):
                os.makedirs(directory_png)
            #print(directory_png)

            #print('title', title)
            CGR_HAPLOTYPES = CGR_HAPLOTYPE_4POINTS("", list()).build_cgr(codify_haplotype, title + peak, directory_png, True) ## --GrayScale
            #FCGR_HAPLOTYPE = FCGR_GS("", 0, list(), list()).build_fcgr(codify_haplotype, k, title, directory_png, True) ## --GrayScale
            #print(FCGR_HAPLOTYPE.get_kmer_freq())
            #print(len(FCGR_HAPLOTYPE.get_kmers_gs()))
            #print(CGR_HAPLOTYPES.get_cgr_seq())

            with open(key+peak+ '.txt', 'w') as output:
                output.write(str(CGR_HAPLOTYPES.get_cgr_seq()))
            break
               



    
