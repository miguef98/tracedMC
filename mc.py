#import open3d as o3d
import numpy as np

def sdf( x ):
    return np.linalg.norm( x, axis=1 )[...,None] - 0.5

def updateDict( hitPositions, N, axis, edgeVertexs, cubeData, polygons ):
    # grid indexed from (back, left, down) to (front, right, up)

    indexs = np.floor( (hitPositions + 1) * ((N-1)/2) ).astype(np.uint32)
    #indexs = (hitPositions + 1) * ((N-1)/2)
    for mainCubeIndex, edgeVertex in zip(indexs, hitPositions):
        edgeVertexIndex = len(edgeVertexs)
        edgeVertexs.append(edgeVertex)

        if axis == 0:
            offsets = np.array([[0,0,0], [0,-1,0],[0,0,-1],[0,-1,-1]])
        
        if axis == 1:
            offsets = np.array([[0,0,0], [-1,0,0],[0,0,-1],[-1,0,-1]])

        if axis == 2:
            offsets = np.array([[0,0,0], [-1,0,0],[0,-1,0],[-1,-1,0]])
        
        polygon = []
        for offset in offsets:
            cubeIndex = mainCubeIndex + offset
            if np.all( np.logical_and( (cubeIndex >= 0), (cubeIndex < N)) ):
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

def rayMarch( rayPositions, axis, df, N, outDict, edgeVertexs, polygons, surfaceThresh = 1e-5 ):
    aliveRays = np.ones( N**2, dtype=bool )

    while np.sum(aliveRays) > 0:
        distances = np.abs( df(rayPositions) )
        rayPositions[ ..., axis ][ ..., None ] += distances
        aliveRays *= rayPositions[ ..., axis ] < 1

        surfaceHitRays = (distances.flatten() < surfaceThresh) * aliveRays
        surfaceHitPosition = rayPositions[ surfaceHitRays ]

        if np.sum(surfaceHitRays) > 0:
            updateDict( 
                surfaceHitPosition,
                N,
                axis,
                edgeVertexs,
                outDict,
                polygons
            )
            rayPositions[surfaceHitRays, axis] =  np.ceil( (surfaceHitPosition[:, axis] + 1) * (N/2) ).astype(np.uint32) * (2/N) - 1

def cubeVertexs( N, df, surfaceThresh ):
    edgeVertexs = []
    polygons = []
    cubeData = {}
    for axis in range(3):
        rayPositions = genGrid( N, axis )
        rayMarch(rayPositions, axis, df, N, cubeData, edgeVertexs, polygons, surfaceThresh=surfaceThresh )

    return cubeData, np.array(edgeVertexs), polygons

def cubeCenterVertex( cubeData, edgeVertexs ):
    for cubeIdx, cubeInfo in cubeData.items():
        P = np.array( edgeVertexs[cubeInfo['edgeVertexIndexs']] )
        A = P / np.linalg.norm(P, axis=1)[...,None]

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
    d, ev, p = cubeVertexs( N, sdf, surfaceThresh=1e-5 )
    cubeCenterVertex( d, ev )
    
    return generateSurface( d, p )

if __name__ == '__main__':
    #print( len( list( cubeVertexs( 32, sdf ).keys() ) ))
    #print( np.concatenate( np.array(np.meshgrid( np.arange(3), np.arange(3), indexing='ij'))[...,None], axis=2 ))
    #print(genGrid( 3, 2 ))
    #quit()
    d, ev, p = cubeVertexs( 4, sdf, surfaceThresh=1e-5 )
    cubeCenterVertex( d, ev )
    
    v,t = generateSurface( d, p )

    #print( len(list(d.keys()) ))

