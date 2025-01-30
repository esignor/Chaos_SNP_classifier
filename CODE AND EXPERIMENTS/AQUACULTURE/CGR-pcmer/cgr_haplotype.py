# reference for develop CGR "Chaos game representation and its applications in bioinformatics", Hannah Franziska Löchel, Dominik Heider

import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import VIRUSES
import AQUACULTURE
from AQUACULTURE.module import *
from VIRUSES.coord import Coord


# [tuple(x_coord, y_coord, cgr_point)]
class CGR_HAPLOTYPE_3POINTS:
  def __init__(self, seq = 'N', cgr_seq = list()): # cgr_list = list(tuple(['N', Coord().get_coords()])
    self.seq = seq #'N' seq for initial point
    self.cgr_seq = cgr_seq
    self.cgr_seq.append(Coord(0,0).get_coords()) # initial point for each cgr (x = 0, y = 0)


  def get_seq(self):
    return self.seq
  
  def get_cgr_seq(self):
    return self.cgr_seq
  
  def __set_seq(self, new_seq):
    self.seq = new_seq

  def __set_cgr_seq(self, coords):
    self.cgr_seq.append(coords)

  # Design the CGR in file .png
  def __design_cgr(self, title, directory):
    fig, (axes) = plt.subplots(1, 1, figsize=(20, 5))
    fig.suptitle(title)
    coord_x = list(); coord_y = list()
    for i in range(0,len(self.cgr_seq)):
      coord_x.append(self.cgr_seq[i][0]); coord_y.append(self.cgr_seq[i][1])
  
    axes.plot(coord_x, coord_y, 'k.', markersize=1)
    axes.set_aspect('equal', adjustable='box')
    axes.axis('off')
    plt.savefig(directory + '/' + 'CGR' + title)
    #plt.clf(); 
    plt.close()

    ## image in GS format
    img = Image.open(directory + '/' + 'CGR' + title + '.png').convert('L')
    img.save(directory + '/' + 'CGR' + title + '.png')

    

  # Apply CGR algorithms
  def build_cgr(self, seq, title = None, directory = None, flag_design = False):
    n = len(seq)
    Coords = np.zeros((2, n+1))
    Coords[:, 0] = np.array([0, 0]) # initial point
    corner_haplotype = Coords[:, 0]
    for i in range(1, n+1):
      if seq[i-1] == '0':
        corner_haplotype = np.array([-1, -1])
      elif seq[i-1] == '1':
        corner_haplotype = np.array([1, -1])
      elif seq[i-1] == '2':
        corner_haplotype = np.array([0, 1])
      Coords[:, i] = Coords[:, i-1] + 0.5 * (corner_haplotype - Coords[:, i-1])

      f_xy = Coord(Coords[0,i], Coords[1,i])

      self.__set_seq(self.seq + seq[i-1])
      self.__set_cgr_seq(f_xy.get_coords())

    if flag_design == True:
      self.__design_cgr(title, directory)

    return self
  


 # [tuple(x_coord, y_coord, cgr_point)]
class CGR_HAPLOTYPE_4POINTS:
  def __init__(self, seq = 'N', cgr_seq = list()): # cgr_list = list(tuple(['N', Coord().get_coords()])
    self.seq = seq #'N' seq for initial point
    self.cgr_seq = cgr_seq
    self.cgr_seq.append(Coord(0,0).get_coords()) # initial point for each cgr (x = 0, y = 0)


  def get_seq(self):
    return self.seq
  
  def get_cgr_seq(self):
    return self.cgr_seq
  
  def __set_seq(self, new_seq):
    self.seq = new_seq

  def __set_cgr_seq(self, coords):
    self.cgr_seq.append(coords)

  # Design the CGR in file .png
  def __design_cgr(self, title, directory):
    fig, (axes) = plt.subplots(1, 1, figsize=(20, 5)) # figsize=(5,5) for vs 2.0
    #fig.suptitle(title)
    coord_x = list(); coord_y = list()
    for i in range(0,len(self.cgr_seq)):
      coord_x.append(self.cgr_seq[i][0]); coord_y.append(self.cgr_seq[i][1])
  
    axes.plot(coord_x, coord_y, 'k.', markersize=1)
    axes.set_aspect('equal', adjustable='box') # only in  vs 1.0
    axes.axis('off')
    #axes.set_title(title)
    #plt.margins(x=0, y=0) # for eliminate the image box (vs 2.0)
    #plt.savefig(directory + '/' + 'CGR' + title + '.png', pad_inches = 0) # vs 2.0.
    plt.savefig(directory + '/' + 'CGR' + title + '.png')
    #plt.clf();  
    plt.close()

    ## image in GS format
    img = Image.open(directory + '/' + 'CGR' + title + '.png').convert('L')
    img.save(directory + '/' + 'CGR' + title + '.png')

    

  # Apply CGR algorithms
  def build_cgr(self, seq, title = None, directory = None, flag_design = False):
    n = len(seq)
    Coords = np.zeros((2, n+1))
    Coords[:, 0] = np.array([0, 0]) # initial point
    corner_haplotype = Coords[:, 0]
    for i in range(1, n+1):
      if seq[i-1] == "0":
        corner_haplotype = np.array([-1, 1])
      elif seq[i-1] == "1":
        corner_haplotype= np.array([-1, -1])
      elif seq[i-1] == "2":
        corner_haplotype = np.array([1, 1])
      elif seq[i-1] == "-1":
        corner_haplotype = np.array([1, -1])
      Coords[:, i] = Coords[:, i-1] + 0.5 * (corner_haplotype - Coords[:, i-1])

      f_xy = Coord(Coords[0,i], Coords[1,i])

      self.__set_seq(self.seq + seq[i-1])
      self.__set_cgr_seq(f_xy.get_coords())

    if flag_design == True:
      self.__design_cgr(title, directory)

    return self
  
