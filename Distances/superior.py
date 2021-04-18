import numpy as np

'''
    In mathematics, Chebyshev distance (or Tchebychev distance), maximum metric, 
    or L∞ metric is a metric defined on a vector space where 
    the distance between two vectors is the greatest of their differences 
    along any coordinate dimension.[2] It is named after Pafnuty Chebyshev.

    It is also known as chessboard distance, since in the game of chess the minimum number of 
    moves needed by a king to go from one square on a chessboard to another equals the 
    Chebyshev distance between the centers of the squares, if the squares have side length one, 
    as represented in 2-D spatial coordinates with axes aligned to the edges of the board.
'''

objA = [22, 1, 42, 10]

objB = [20, 0, 36, 8]

npA = np.array(objA)

npB = np.array(objB)

chebyshev = np.abs(npA - npB).max()

# chebyshev = np.linalg.norm(npA -npB, ord=np.inf)

print(chebyshev)