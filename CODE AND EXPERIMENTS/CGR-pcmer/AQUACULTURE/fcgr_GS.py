import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import VIRUSES
import AQUACULTURE

from AQUACULTURE.module import *
from AQUACULTURE.cgr_haplotype import CGR_HAPLOTYPE_4POINTS
from VIRUSES.fcgr import FCGR
from AQUACULTURE.functions_FCGR_AQUACULTURE import count_kmers,  ratioFreq, preprocessing_line

class FCGR_GS(FCGR):
    def __init__(self, seq = "", kmer = 0, kmer_freq = list(), kmers_gs= list()):
        FCGR.__init__(self, seq = "", kmer = 0, kmer_freq = list())
        self.kmers_gs = kmers_gs

    def __set_kmers_gs(self, kmers_gs):
        self.kmers_gs.append(kmers_gs)

    def get_kmers_gs(self):
        return self.kmers_gs
    

    def __design_fcgr(self, title, directory):
        len_square = int(math.sqrt(2**(2*self.kmer)))
        new_list = []; list_gs = []
        count = 0; freq = self.kmer_freq; kmer_gs = self.kmers_gs
        
        for i in freq:
            count += 1; kmer = i[0] ## fcgr
            if kmer == 0: list_gs.append(0) 
            else:
                for j in kmer_gs: ## gs
                    if kmer == j[0]: list_gs.append(j[1]); break


            if count == len_square: new_list.append(list_gs); count = 0; list_gs = []
        grid = np.array(new_list, dtype=np.uint8)

        im = Image.fromarray(grid).resize((1600,1600), resample=Image.NEAREST)

        plt.axis('off')
        plt.imshow(im)
        title = title.replace(' ',"")
        plt.savefig(directory + '/' + 'FCGR' + title + "(k = "+ str(self.kmer) + ")")
        plt.clf(); plt.close()

        # save in GS format
        img = Image.open(directory + '/' + 'FCGR' + title + "(k = "+ str(self.kmer) + ").png").convert('L')
        img.save(directory + '/' + 'FCGR' + title + "(k = "+ str(self.kmer) + ").png")
    
        return self
    



    def frequency_kmersGS(self):
        maxFreq = self.max_freq()
        ratio_freq = ratioFreq(maxFreq, 255) 

        for entry in self.kmer_freq:
            kmer = entry[0]
            freq = entry[1]

            gs_channel = freq * ratio_freq
   

            self.__set_kmers_gs([kmer, round(gs_channel)])

        return self
    

    def build_fcgr(self, seq, k, title = None, directory = None, flag_design = False):  
        frequency = count_kmers(seq, k)
        array_size = int(math.sqrt(2**(2*k))) # cells for row/colomn present in the fcgr bi-dimensional matrix
        super().init_seq_kmer(array_size)

        matrix_fcgr = []; seq_fcgr = []
        for i in range(array_size):
          matrix_fcgr.append([0]*array_size)
          seq_fcgr.append([0]*array_size)

        for seq_kmer, freq in frequency.items():
          seq_kmer =  preprocessing_line(str(seq_kmer))
          CGR_tmp = CGR_HAPLOTYPE_4POINTS('N', list()).build_cgr(seq_kmer)
          cgr_seq = CGR_tmp.get_cgr_seq()
          x = cgr_seq[k][0]; y = cgr_seq[k][1] # coordinates (x,y) in fcgr for the k-mers
          x_supp = -1; y_supp = -1
          i = -1; j = array_size
          ratio_cell = 2 / array_size
    
          
          # calculate the exact x position in fcgr matrix for the k-mers (sequence_data)
          while x_supp <= x:
            x_supp += ratio_cell
            i += 1 # 0..array_size-1
          # calculate the exact y postion in fcgr matrix for the k-mers (sequence_data)
          while y_supp <= y:
            y_supp += ratio_cell
            
            j -= 1 # array_size-1..0 (i.e., 'AA': y_coord = 3 and y = 1; 0.5+0.5+0.5+0.5, y=4-1-1-1=3)

          matrix_fcgr[j][i] = freq # [j][i] coord y, x = riga j, colonna i
          seq_fcgr[j][i] = seq_kmer

          index = j * array_size + i
          self.set_fcgr_intofcgr_pcmer_rgb(seq, k, index, seq_kmer, freq)


        self.frequency_kmersGS()
        
        if flag_design ==  True:
            self.__design_fcgr(title, directory)
        
        return self
     
    
    
