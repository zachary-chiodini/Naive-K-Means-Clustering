import numpy as np
from random import randint
from nptyping import NDArray

class KMeans( object ) :
    '''
    Naive K-Means Clustering
    '''
    def __init__(
        self : object,
        X : NDArray[ NDArray[ float ] ],
        k : int,
        ) -> None :
        self.X = X     # input matrix
        self.k = k     # number of clusters
        self.sse = 0.0 # sum of squared errors
        self.C = np.array([])
        self.result = {}
        

    def classify( self : object,
                  repeat : int = 1 ) -> None :
        '''
        Classifies Each Row in X
        '''
        results = {}
        centroids = {}
        for i in range( repeat ) :
            # ith k means attempt
            self.__genCentroids()
            Cprev = np.empty( self.C.shape )
            while not np.array_equal( Cprev, self.C ) :
                Cprev = np.copy( self.C )
                self.__update()
            # store results of ith attempt
            results[ self.sse ] = self.result
            centroids[ self.sse ] = self.C
            # reset results
            self.result = {}
            self.C = np.array([])
            self.sse = 0.0
        # choose centroids and
        # result with minimum SSE
        indx = min( results.keys() )
        self.sse = indx
        self.C = centroids[ indx ]
        self.result = results[ indx ]
        return

    def __genCentroids( self : object ) -> None :
        '''
        Generates K Centroids
        '''
        prev = []
        self.C = np.array([]).reshape( 0, self.X[ 0 ].size )
        for i in range( self.k ) :
            indx = randint( 0, len( self.X ) - 1 )
            while indx in prev :
                indx = randint( 0, len( self.X ) - 1 )
            self.C = np.vstack( [ self.C, self.X[ indx ] ] )
            prev.append( indx )
        return

    def __update( self : object ) -> None :
        '''
        Updates Each Row in C and
        Each Classification in X
        '''
        self.sse = 0.0
        for i in range( 1, self.k + 1 ) :
            M = np.prod(
                [ np.square( self.X - self.C[ i - 1 ] ).sum( axis = 1 ) <=\
                  np.square( self.X - self.C[ j % self.k ] ).sum( axis = 1 )
                  for j in range( i, i + self.k - 1 ) ],
                axis = 0,
                dtype = bool
                )
            self.C[ i - 1 ] = self.X[ M ].sum( axis = 0 ) / M.sum() \
                              if M.sum() else self.C[ i - 1 ]
            self.result[ i ] = self.X[ M ]
            self.sse += np.square( self.X[ M ] - self.C[ i - 1 ] ).sum()
        return
