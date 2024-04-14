import open3d as o3d
import open3d.core as o3c
import numpy as np
from scipy.spatial import KDTree

def df( x ):
    # esfera
    return np.linalg.norm( x, axis=1 )[...,None] - 0.5

    # spot numerico
    return spotScene.compute_distance( o3c.Tensor(x, dtype=o3c.float32) ).numpy()[...,None]


def g_df( x ):

    # esfera
    return x / np.linalg.norm( x, axis=1 )[..., None] * np.random.choice( [-1,1], (x.shape[0],1))

    # spot numerico
    EPS = 0.001
    gradient = ( np.array( [ 
        df( x + np.tile( dx, (len(x),1)) ) - df( x - np.tile( dx, (len(x),1)))
        for dx in np.array( [[ EPS, 0,0 ], [0, EPS,0 ], [0,0,EPS]] )
    ] ) / (2* EPS) ).reshape( x.shape )

    return gradient / np.linalg.norm( gradient, axis=1 )[..., None]


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
    #indexs = np.floor( (hitPositions + 1) * ((N-1)/2) ).astype(np.uint32)
    #indexs = (hitPositions + 1) * ((N-1)/2)
    for mainCubeIndex, edgeVertex, edgeNormal in zip(indexs, hitPositions, hitNormals):
        edgeVertexIndex = len(edgeVertexs)
        edgeVertexs.append(edgeVertex)
        edgeNormals.append(edgeNormal)

        if axis == 0:
            offsets = np.array([[0,0,0], [0,-1,0],[0,0,-1],[0,-1,-1]])
        
        elif axis == 1:
            offsets = np.array([[0,0,0], [-1,0,0],[0,0,-1],[-1,0,-1]])

        elif axis == 2:
            offsets = np.array([[0,0,0], [-1,0,0],[0,-1,0],[-1,-1,0]])
        
        polygon = []
        for offset in offsets:
            cubeIndex = mainCubeIndex + offset
            if np.all( np.logical_and( (cubeIndex >= 0), (cubeIndex < (N-1))) ):
                polygon.append( tuple(cubeIndex) )
                if tuple(cubeIndex) in cubeData:
                    cubeData[tuple(cubeIndex)]['edgeVertexIndexs'].append(edgeVertexIndex)
                else:
                    cubeData[tuple(cubeIndex)] = { 'edgeVertexIndexs':[ edgeVertexIndex ] }

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
            rayPositions[surfaceHitRays, axis] =  (np.floor( (surfaceHitPosition[:, axis] + 1) * ((N-1)/2) ) + 1) * (2/(N-1)) - 1

def cubeVertexs( N, df, surfaceThresh ):
    edgeVertexs = []
    edgeNormals = []
    polygons = []
    cubeData = {}
    for axis in range(3):
        rayPositions = genGrid( N, axis )
        rayMarch(rayPositions, axis, df, N, cubeData, edgeVertexs, edgeNormals, polygons, surfaceThresh=surfaceThresh )

    return cubeData, np.array(edgeVertexs),np.array(edgeNormals), polygons

def cubeCenterVertex( cubeData, edgeVertexs, edgeNormals ):
    for cubeIdx, cubeInfo in cubeData.items():
        P = np.array( edgeVertexs[cubeInfo['edgeVertexIndexs']] )
        A = np.array( edgeNormals[cubeInfo['edgeVertexIndexs']] )

        if len(A) == 1:
            cubeData[cubeIdx]['vertex'] = P.flatten()
        elif len(A) == 2:
            cubeData[cubeIdx]['vertex'] = (P.sum(0) / 2).flatten()
        else:
            b = np.sum( A * P, axis=1).flatten()
            x,_,_,_ = np.linalg.lstsq( A, b, rcond=None )
            #x = np.linalg.solve( A.T @ A, A.T @ b )

            cubeData[cubeIdx]['vertex'] = x.flatten()

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
            t2 = [ getVertexIndex( cubeIndex, cubeData, vertexIndexs, vertices ) for cubeIndex in reversed(polygon[1:])]


            triangles.append(t1)
            triangles.append(t2)
        elif len(polygon) == 3:
            triangles.append([ getVertexIndex( cubeIndex, cubeData, vertexIndexs, vertices ) for cubeIndex in polygon])
        else:
            # no agrego poligonos de 2 o menos vertices
            continue

    return np.array(vertices), np.array(triangles)

def tracedMarchingCubes( N, surfaceThresh=1e-5 ):
    d, ev, en, p = cubeVertexs( N, df, surfaceThresh=1e-5 )
    cubeCenterVertex( d, ev, en )
    
    return generateSurface( d, p )

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

