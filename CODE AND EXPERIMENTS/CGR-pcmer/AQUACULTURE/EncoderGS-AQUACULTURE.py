## MODULE
import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE

from AQUACULTURE.module import * 
from AQUACULTURE.cgr_haplotype import CGR_HAPLOTYPE_4POINTS
#from AQUACULTURE.cgr_haplotype import CGR_HAPLOTYPE_3POINTS # for haplotype CGR with 3 points 
from AQUACULTURE.functions_FCGR_AQUACULTURE import parse_mortality_flag, read_feature_list, parse_haplotypes_codify

def img_setCluster(haplotypes_file, data, mortality_data, out_directory, path_datasets, cl):
        cluster = pd.read_csv(path_datasets + '/cluster-' + str(cl) + '.csv', header=None, skiprows=1, low_memory=False)
        print('Open cluster' + str(cl))
        selected_features = read_feature_list(cluster)
        print('Extracted features in cluster ' + str(cl))
        
        print('len',len( selected_features))
        for n in range(0, len(selected_features)):
            id_feature = selected_features[n]
            title = id_feature; #print('title', title)
       
            flag_mortality = parse_mortality_flag(mortality_data, id_feature)
            print('Extreacted mortality flag', type(flag_mortality), flag_mortality)
            if flag_mortality == 0: dir_mortality = "ALIVE" # vivo
            elif flag_mortality == 1: dir_mortality = "DEAD" # morto

            codify_haplotype = parse_haplotypes_codify(data, id_feature)
            
            print('Extracted haplotype codify', codify_haplotype, type(codify_haplotype), len(codify_haplotype))
        
            directory_png = out_directory + '/CGR/' + 'GS 4 points/' +  haplotypes_file + '/' + str(cl) + '/' + dir_mortality
            if not os.path.exists(directory_png): os.makedirs(directory_png, exist_ok=True)
            print(directory_png + '/' + title)

            CGR_HAPLOTYPES = CGR_HAPLOTYPE_4POINTS("", list()).build_cgr(codify_haplotype, title, directory_png, True) ## --GrayScale



        

if __name__ == '__main__':
    
    
    haplotypes_file = "features-active50"

    MAIN_FOLDER = 'CODE AND EXPERIMENTS'
    out_directory = MAIN_FOLDER + '/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/include-chr3'
    path_datasets = MAIN_FOLDER + '/DATASET/PNRR'

    mortality_data = pd.read_csv(path_datasets + '/mortality.csv')
    print('Open mortality')
    data = pd.read_csv(path_datasets + '/' + haplotypes_file + '.csv', header=None, low_memory=False)
    print('Ok extracted datasets')
    

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)


    t1 = threading.Thread(target = img_setCluster, args = (haplotypes_file, data, mortality_data, out_directory, path_datasets, 0))
    t2 = threading.Thread(target = img_setCluster, args = (haplotypes_file, data, mortality_data, out_directory, path_datasets, 1))
    t1.start(); t2.start()
    print('Finished threads')
               



    
