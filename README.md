# CNN_protein_prediction

Description: Convolutional neural network for predicting the secondary structure of a protein based on its amino acid 
sequence (in progress)

Input: A protein, represented by its amino acid sequence. Each amino acid is represented by a row in the matrix containing
a one-hot encoding over the 20 possible amino acid identies, plus 20 additional indices with probability information that
that particular amino acid identity would occuring at that particular position in a protein (based on evolutionary info).

Output: The secondary structure of the protein, i.e. a series of classifications of how sections of the protein fold. Helix (H)
Sheet (E) or Coil (C)

Model: A CNN with three convolutional layers implemented with Tenserflow. 

To run: $ python cnn.py
