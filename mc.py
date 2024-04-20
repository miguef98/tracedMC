import numpy as np
from dudf.model import SIREN
from dudf.evaluate import evaluate
from scipy.optimize import minimize
import torch
import networkx as nx

model = SIREN(
            n_in_features= 3,
            n_out_features=1,
            hidden_layer_config=[256]*8,
            w0=30,
            ww=None,
            activation='sine'
)
model.load_state_dict( torch.load('models/spot_smooth/model_best.pth'))
device_torch = torch.device(0)
model.to(device_torch)

def df( x ):
    # spot dudf
    return evaluate( model, x, device=device_torch)
    
    # esfera
    return np.linalg.norm( x, axis=1 )[...,None] - 0.5

    # spot numerico
    return spotScene.compute_distance( o3c.Tensor(x, dtype=o3c.float32) ).numpy()[...,None]


def g_df( x ):
    # dudf
    gradients = np.zeros_like(x)
    hessians = np.zeros((x.shape[0], 3, 3))
    pred_distances = evaluate( model, x, device=device_torch, gradients=gradients, hessians=hessians )
    eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(hessians) )
    pred_normals = eigenvectors[..., 2].numpy()

    return pred_normals

    # esfera
    return x / np.linalg.norm( x, axis=1 )[..., None] * np.random.choice( [-1,1], (x.shape[0],1))



    # spot numerico
    EPS = 0.001
    gradient = ( np.array( [ 
        df( x + np.tile( dx, (len(x),1)) ) - df( x - np.tile( dx, (len(x),1)))
        for dx in np.array( [[ EPS, 0,0 ], [0, EPS,0 ], [0,0,EPS]] )
    ] ) / (2* EPS) ).reshape( x.shape )

    return gradient / np.linalg.norm( gradient, axis=1 )[..., None]

def ransac( X, A, b ):
    if A.shape[0] == 3:
        return leastSquares(X, A,b)
    else:
        x_best = None
        error_best = np.inf
        for i in range(20):
            subsetIdx = np.random.choice(A.shape[0], np.random.randint(3, A.shape[0]))
            A_bar = A[ subsetIdx , : ]
            X_bar = X[ subsetIdx , : ]
            b_bar = b[ subsetIdx ]

            x_bar = leastSquares( X_bar, A_bar, b_bar )
            error_bar = np.linalg.norm( A_bar @ x_bar - b_bar )

            if error_best > error_bar:
                x_best = x_bar
                error_best = error_bar

        return x_best            

def leastSquares( X, A, b, threshold=0.1 ):
    U,S,V_t = np.linalg.svd( A )
    reliableSVs = S > threshold
    numberSVs = np.sum(reliableSVs)
    S_inv = np.where( reliableSVs, np.divide( np.ones_like(S), S) , np.zeros_like(S) )
    
    def solParticular():
        if A.shape[0] > A.shape[1]:
            A_pseudoinv = V_t.T @ np.block( [ np.diag( S_inv ), np.zeros( (S_inv.shape[0], U.T.shape[0] - S_inv.shape[0])) ] ) @ U.T
        elif A.shape[0] < A.shape[1]:
            A_pseudoinv = V_t.T @ np.vstack( [ np.diag( S_inv ), np.zeros( (V_t.shape[0] - S_inv.shape[0], S_inv.shape[0])) ] ) @ U.T
        else:
            A_pseudoinv = V_t.T @ np.diag( S_inv ) @ U.T

        return A_pseudoinv @ b

    if numberSVs == 0:
        return np.mean(X, axis=0)

    elif numberSVs == 1:
        c = solParticular()
        v1 = V_t[1,:]
        v2 = V_t[2,:]
        x = np.mean(X, axis=0)

        alpha1 = (x - c) @ v1
        alpha2 = (x - c) @ v2

        return c + alpha1 * v1 + alpha2 * v2

    elif numberSVs == 2:
        # solGeneral = solParticular() + alpha * V_t[2,:]
        c = solParticular()
        v = V_t[2,:]
        x = np.mean(X, axis=0)
        return c + ((v @ (x - c)) / (v @ v) ) * v

    else:
        return solParticular()

def dualContour( X, A, b, threshold ):
    # paper
    return leastSquares( X, A, b, threshold=threshold )

    # paper + ransac
    return ransac(X, A, b )

    #center of mass
    return np.mean(X, axis=0)


    # recta plano
    planeParams = leastSquares( np.hstack([X, np.ones((X.shape[0],1)) ]), np.zeros(X.shape[0]) )
    planeNormal = planeParams[:3]
    centerOfMass = np.sum(X, axis=0) / X.shape[0]

    lambdaMin = (planeNormal.T @ A.T @ b - planeNormal.T @ A.T @ A @ centerOfMass) / (planeNormal.T @ A.T @ A @ planeNormal)

    return lambdaMin * planeNormal + centerOfMass

    # recta plano v2
    planeNormal = np.sum(A, axis=0) / A.shape[0] # que pasa si las normales miran opuestas ?
    planeNormal /= np.linalg.norm(planeNormal)
    centerOfMass = np.sum(X, axis=0) / X.shape[0]

    lambdaMin = (planeNormal.T @ A.T @ b - planeNormal.T @ A.T @ A @ centerOfMass) / (planeNormal.T @ A.T @ A @ planeNormal)

    return lambdaMin * planeNormal + centerOfMass

    # equidistancia con minimizacion
    def f(x, epsilon):
        v = x @ x.T * np.ones( X.shape[0] ) - 2 * x @ X.T + np.sum( X * X, axis=1 ) -  epsilon * np.ones(X.shape[0])
        return v.T @ v

    v1 = minimize( lambda x: f(x, 0.1), (np.sum(X, axis=0) / X.shape[0]) ).x
    v2 = minimize( lambda x: f(x, 0.9), (np.sum(X, axis=0) / X.shape[0]) ).x

    if np.isclose(np.linalg.norm(v2 - v1), 0):
        return np.sum(X, axis=0)

    planeNormal = (v2 - v1) / np.linalg.norm(v2 - v1)
    centerOfMass = v1
    
    # equidistancia con CML
    epsilon = 0.01
    A_lst = -2 * X
    b_lst = np.sum( X * X, axis=1 ) -  epsilon * np.ones_like(b)

    return leastSquares( A_lst, b_lst )

    # double energy
    lambda1 = 0 #0.01
    lambda2 = 1
    C = np.hstack( [ np.ones((X.shape[0], 1)), np.zeros((X.shape[0], X.shape[0]-1))] ) - np.eye(X.shape[0], X.shape[0])
    A2 = C @ X
    b2 = np.diag( X @ X.T ) / 2

    A = np.vstack([ np.sqrt(lambda1) * A, np.sqrt(lambda2) * A2 ])
    b = np.vstack([ np.sqrt(lambda1) * b[...,None], np.sqrt(lambda2) * b2[...,None] ]).flatten()
    
    # bounded optimization
    E =  lambda x: (A @ x - b).T @ (A @ x - b)
    return minimize( E, x0, bounds=bounds ).x

    # numpy
    x,_,_,sv = np.linalg.lstsq( A, b, rcond=None )
    return x.flatten()

def getCubeIndexs( hitPositions, axis, rayIndexs, N ):
    index = np.floor( (hitPositions[:, axis] + 1) * ((N-1)/2) ).astype(np.uint32)
    
    if axis == 0:
        indexs = np.concatenate( [ index[...,None], rayIndexs[:,0][...,None], rayIndexs[:,1][...,None] ], axis=1 )
    elif axis == 1:
        indexs = np.concatenate( [ rayIndexs[:,0][...,None], index[...,None], rayIndexs[:,1][...,None] ], axis=1 )
    elif axis == 2:
        indexs = np.concatenate( [ rayIndexs[:,0][...,None], rayIndexs[:,1][...,None], index[...,None] ], axis=1 )

    return indexs

def updateDict( hitPositions, hitNormals, rayIndexs, N, axis, edgeVertexs, edgeNormals, cubeData, polygons ):
    # grid indexed from (back, left, down) to (front, right, up)
    indexs = getCubeIndexs( hitPositions, axis, rayIndexs, N )

    for mainCubeIndex, edgeVertex, edgeNormal in zip(indexs, hitPositions, hitNormals):
        edgeVertexIndex = len(edgeVertexs)
        edgeVertexs.append(edgeVertex)
        edgeNormals.append(edgeNormal)

        if axis == 0:
            offsets = np.array([[0,0,0], [0,-1,0],[0,0,-1],[0,-1,-1]])
            edges = [ 3, 1, 7, 5 ]
        
        elif axis == 1:
            offsets = np.array([[0,0,0], [-1,0,0],[0,0,-1],[-1,0,-1]])
            edges = [ 0, 2, 4, 6 ]

        elif axis == 2:
            offsets = np.array([[0,0,0], [-1,0,0],[0,-1,0],[-1,-1,0]])
            edges = [ 8, 9, 11, 10 ]
        
        polygon = []
        for offset, edge in zip( offsets, edges):
            cubeIndex = mainCubeIndex + offset
            if np.all( np.logical_and( (cubeIndex >= 0), (cubeIndex < (N-1))) ):
                polygon.append( tuple(cubeIndex) )
                if tuple(cubeIndex) in cubeData:
                    if edge in cubeData[tuple(cubeIndex)]['edgeVertexIndexs']:
                        raise Exception('ups')
                        cubeData[tuple(cubeIndex)]['edgeVertexIndexs'][edge].append(edgeVertexIndex)
                    else:
                        cubeData[tuple(cubeIndex)]['edgeVertexIndexs'][edge] = [ edgeVertexIndex ]
                else:
                    cubeData[tuple(cubeIndex)] = { 'edgeVertexIndexs': { edge: [ edgeVertexIndex ] } } 

        polygons.append(polygon)

def genGrid( N, axis ):
    if axis == 0:
        xs = np.ones( (N,N) ) * -1
        ys, zs = np.meshgrid( np.linspace(-1,1,N),np.linspace(-1,1,N))
    elif axis == 1:
        ys = np.ones( (N,N) ) * -1
        xs, zs = np.meshgrid( np.linspace(-1,1,N),np.linspace(-1,1,N))
    elif axis == 2:
        zs = np.ones( (N,N) ) * -1
        xs, ys = np.meshgrid( np.linspace(-1,1,N),np.linspace(-1,1,N))
    else: 
        raise ValueError('Invalid axis')
    
    return np.concatenate( [xs[...,None],ys[...,None],zs[...,None]], axis=2).reshape(N**2, 3)    

def rayMarch( rayPositions, axis, df, N, outDict, edgeVertexs, edgeNormals, polygons, surfaceThresh = 1e-5 ):
    aliveRays = np.ones( N**2, dtype=bool )
    x,y=np.meshgrid( np.arange(N),np.arange(N), indexing='xy' )
    rayIndexs = np.concatenate( [x[...,None], y[...,None]], axis=2 ).reshape(N**2, 2)

    while np.sum(aliveRays) > 0:
        distances = np.abs( df(rayPositions) )
        rayPositions[ ..., axis ][ ..., None ] += distances
        aliveRays = np.logical_and( aliveRays, rayPositions[ ..., axis ] < 1 )

        surfaceHitRays = np.logical_and( distances.flatten() < surfaceThresh, aliveRays)
        surfaceHitPosition = rayPositions[ surfaceHitRays ]

        if np.sum(surfaceHitRays) > 0:
            surfaceHitNormals = g_df( surfaceHitPosition )
            updateDict( 
                surfaceHitPosition,
                surfaceHitNormals,
                rayIndexs[surfaceHitRays],
                N,
                axis,
                edgeVertexs,
                edgeNormals,
                outDict,
                polygons
            )
            
            rayPositions[surfaceHitRays, axis] =  (np.floor( (surfaceHitPosition[:, axis] + 1) * ((N-1)/2) ) + 1 + np.random.uniform(0, 2/(N-1))) * (2/(N-1)) - 1


def cubeVertexs( N, df, surfaceThresh ):
    edgeVertexs = []
    edgeNormals = []
    polygons = []
    cubeData = {}
    for axis in range(3):
        rayPositions = genGrid( N, axis )
        rayMarch(rayPositions, axis, df, N, cubeData, edgeVertexs, edgeNormals, polygons, surfaceThresh=surfaceThresh )

    return cubeData, np.array(edgeVertexs),np.array(edgeNormals), polygons

def cubeCenterVertex( cubeData, edgeVertexs, edgeNormals, N, threshold, alpha=6 ):
    global corners
    for cubeIdx, cubeInfo in cubeData.items():
        X = np.array( [ np.mean(edgeVertexs[ idxPerEdge ], axis=0) for idxPerEdge in cubeInfo['edgeVertexIndexs'].values() ])
        A = np.array( [ np.mean(edgeNormals[ idxPerEdge ], axis=0) for idxPerEdge in cubeInfo['edgeVertexIndexs'].values() ])

        G = X @ X.T
        P = -2 * G + np.outer( np.diag(G), np.ones(X.shape[0]) ) + np.outer( np.ones(X.shape[0]), np.diag(G) )

        X_new = []
        A_new = []
        for components in nx.connected_components( nx.from_numpy_array( P < (2/(alpha*(N-1)) ) ** 2 )):
            X_new.append( np.mean( X[np.array(list(components)) ], axis=0 ))
            A_new.append( np.mean( A[np.array(list(components)) ], axis=0 ))

        X_new = np.array(X_new)
        A_new = np.array(A_new)

        cubeData[cubeIdx]['X'] = X_new
        cubeData[cubeIdx]['A'] = A_new

        if len(A_new) == 1:
            cubeData[cubeIdx]['vertex'] = X_new.flatten()
        else:
            b = np.sum( A_new * X_new, axis=1).flatten()
            dual = dualContour( X_new, A_new, b, threshold=threshold )

            #dual = dualContour( A, b, threshold=threshold )
            #bounds = [ (-1 + i*2/(N-1), -1 + (i+1)*2/(N-1)) for i in cubeIdx ]
            #dual = dualContour( A, b, np.sum(X, axis=0) / X.shape[0], bounds )

            cubeData[cubeIdx]['vertex'] = dual # x.flatten()

def getVertexIndex( cubeIndex, cubeData, vertexIndexs, vertices):
    if cubeIndex not in vertexIndexs:
        vertexIndex = len(vertices)
        vertices.append( cubeData[cubeIndex]['vertex'] )
        vertexIndexs[ cubeIndex ] = vertexIndex

    return vertexIndexs[cubeIndex] 

def generateSurface( cubeData, polygons ):
    vertexIndexs = {}
    vertices = []
    triangles = []

    for polygon in polygons:
        if len(polygon) == 4:
            t1 = [ getVertexIndex( cubeIndex, cubeData, vertexIndexs, vertices ) for cubeIndex in polygon[:3]]
            t2 = [ getVertexIndex( cubeIndex, cubeData, vertexIndexs, vertices ) for cubeIndex in polygon[1:]] #reversed(polygon[1:])]


            triangles.append(t1)
            triangles.append(t2)
        elif len(polygon) == 3:
            triangles.append([ getVertexIndex( cubeIndex, cubeData, vertexIndexs, vertices ) for cubeIndex in polygon])
        else:
            # no agrego poligonos de 2 o menos vertices
            continue

    return np.array(vertices), np.array(triangles)

def tracedMarchingCubes( N, surfaceThresh=1e-5 ):
    d, ev, en, p = cubeVertexs( N, df, surfaceThresh=surfaceThresh )
    cubeCenterVertex( d, ev, en,N )

    v,t = generateSurface( d, p )
    return v,t


#spotMesh = o3d.t.io.read_triangle_mesh('spot.obj')
#spotScene = o3d.t.geometry.RaycastingScene()
#spotScene.add_triangles(spotMesh)

#if __name__ == '__main__':
    #print( len( list( cubeVertexs( 32, sdf ).keys() ) ))
    #print( np.concatenate( np.array(np.meshgrid( np.arange(3), np.arange(3), indexing='ij'))[...,None], axis=2 ))
    #print(genGrid( 3, 2 ))
    #quit()
    #d, ev, p = cubeVertexs( 4, df, surfaceThresh=1e-5 )
    #cubeCenterVertex( d, ev )
    
    #v,t = generateSurface( d, p )

    #print( len(list(d.keys()) ))

