import numpy as np
from random import random
from nptyping import NDArray

class KMeans( object ) :
    '''
    Naive K-Means Clustering
    '''
    def __init__(
        self : object,
        X : NDArray[ NDArray[ float ] ], # input matrix
        k : int,                         # number of clusters
        ) -> None :
        self.X = X
        self.k = k
        self.C = np.array([])
        self.result = {}
        

    def classify( self : object ) -> None :
        '''
        Classifies Each Row in X
        '''
        self.__genCentroids()
        Cprev = np.empty( self.C.shape )
        while not np.array_equal( Cprev, self.C ) :
            Cprev = np.copy( self.C )
            self.__update()
        return

    def __genCentroids( self : object ) -> None :
        '''
        Generates K Centroids
        '''
        self.C = np.array([]).reshape( 0, self.X[ 0 ].size )
        mn = [ self.X[ :, j ].min()
               for j in range( self.X[ 0 ].size ) ]
        mx = [ self.X[ :, j ].max()
               for j in range( self.X[ 0 ].size ) ]
        for i in range( self.k ) :
            ri = np.empty( self.X[ 0 ].shape )
            for j in range( self.X[ 0 ].size ) :
                ri[ j ] = mn[ j ] + ( mx[ j ] - mn[ j ] )*random()
            self.C = np.vstack( [ self.C, ri ] )
        return

    def __update( self : object ) -> None :
        '''
        Updates Each Row in C and
        Each Classification in X
        '''
        for i in range( 1, self.k + 1 ) :
            M = np.prod(
                [ np.square( self.X - self.C[ i - 1 ] ).sum( axis = 1 ) <=\
                  np.square( self.X - self.C[ j % self.k ] ).sum( axis = 1 )
                  for j in range( i, i + self.k - 1 ) ],
                axis = 0,
                dtype = bool
                )
            self.result[ i ] = self.X[ M ]
            self.C[ i - 1 ] = self.X[ M ].sum( axis = 0 ) / M.sum() \
                              if M.sum() else self.C[ i - 1 ]
        return
