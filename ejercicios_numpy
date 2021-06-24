#22 Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
import numpy as np
matriz1=np.ones((5,3))
matriz2=np.ones((3,2))
matrizResultante=np.dot(matriz1,matriz2)
print(matrizResultante)

#23 Given a 1D array, negate all elements which are between 3 and 8, in place.
import numpy as np
matriz = np.arange(10)  ##creacion de una matriz entre 0 al 9
np.putmask(matriz, (3 <matriz) & (matriz<8), matriz* -1)  ##con putmask le damos la propiedad para que se 
#cambien los valores que se necesiten en el arreglo.
print(matriz) ##impresion de la matriz resultante

#34 Consider two random array A anb B, check if they are equal
A = np.random.randint(4,10,5)        # el primer y segundo parametro indican el rango de numeros aleatorios 
#que se permiten estar en el arreglo 
B = np.random.randint(4,10,5)        #el tercer parametro indica el tamaño del arreglo
equal = np.allclose(A,B)             #la funcion allclose sirve para detectar si dos arreglos son iguales
print(A)
print(B)
print("Son iguales? " + str(equal))

#36 Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates 
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
# En la solución primero la matriz Z la separamos por columnas en dos matrices X e Y asignándole todos los 
# valores usando la sentencia
# [:, 0] en el caso de X y [:, 1] en el caso de Y, consecuente a ello calculamos la raíz cuadrada de ambos 
# arreglos elevados al cuadrado
# y, también, la función inversa de tangente.

##45 How to convert a float (32 bits) array into an integer (32 bits) in place?
import numpy as np
lista = [1.3, 4.5, 2.3, 6.4]    ##creacion de una lista de tipo float
matriz = np.array(lista)        ##convirtiendo la lista en una matriz
matriz1 = np.array(matriz, dtype=np.int32)      ## creamos otra matriz a la cual le damos el parametro para 
#que sus valores sean enteros.
print(matriz1)

#46 How to read the following file?
def creartxt():
    archivo=open('data.txt','w')
    archivo.close()
    
def grabartxt():
    archivo=open('data.txt','a') 
    archivo.write('1,2,3,4,5\n') 
    archivo.write('6,,,7,8\n') 
    archivo.write(',,9,10,11\n') 
    archivo.close()

creartxt()
grabartxt()
Z = np.genfromtxt("data.txt", delimiter=",")  #procesa datos con más velocidad
print(Z)


#54 Create an array class that has a name attribute
class NamedArray(np.ndarray):  #definiendo la clase
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls) #añadiendo un objeto tipo array 
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")
Z = NamedArray(np.arange(10), "range_10")
print (Z.name) #Impresion del rango de la matriz

#60 How to get the diagonal of a dot product? 
print(np.dot(A, B))
#En este ejemplo se realiza primero la multiplicación de matrices para luego hacer el producto diagonal.
print(np.sum(A + B.T, axis=1))
#En esta versión realiza lo mismo que nparray.sum, sin embargo, esta nos devuelve una matriz, en decir se 
# realiza la suma a lo largo de un eje dado. 
print(np.einsum("ij,ji->i", A, B))
#En esta versión Podemos controlar la salida con un identificador -> y aumenta la flexibilidad de la función, 
# además de controlar la suma, nos devuelve directamente 
# una multiplicación de matrices, además, optimizamos el uso de memoria.

#63 How to swap two rows of an array?
A = np.arange(25).reshape(5,5)   #reshape sirve para transformar una arreglo en en una matriz, en este caso de 5x5.
print("Arreglo A[[1,0]]:")
print(A[[1,0]])         # A[[1,0]] funciona como un nuevo arreglo pero sin serlo, en donde el orden en el que se presenta 
                        # es: la fila numero 1 de A en la posicion 0 de este subarreglo y la fila 0 en la posicion 1.
A[[0,1]] = A[[1,0]]     # El subarreglo A[[1,0]] se le asigna al sub arreglo A[[0,1]] reemplazando en el arreglo original 
#la fila 0 y 1 
print("Arreglo modificado")
print(A)

#64 Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments 
# composing all the triangles
faces = np.random.randint(0,100,(10,3)) #Creacion de matriz de 10x3 con numeros aleatorios del 0 al 100
print()
print(faces)
print()
print(faces.repeat(2,axis=1))

F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
print()
print(F)
F = F.reshape(len(F)*3,2)
print()
print(F)
F = np.sort(F,axis=1)
print()
print(F)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
print()
print(G)
G = np.unique(G)
print(G)

#65 Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?
import numpy as np
C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)

#69 Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])?
def distance(P0,P1,p):    
    T=P1-P0
    L=(T**2).sum(axis=1)
    U= -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U=U.reshape(len(U),1)             # se determinan las filas y el sistema de columnas las determina automáticamente
    D=P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))
P0=np.random.uniform(-10,10,(10,2))   # genera un número decimal entre a y b que será el conjunto de puntos P0 
P1=np.random.uniform(-10,10,(10,2))   # genera un número decimal entre a y b que será el conjunto de puntos P1 
p=np.random.uniform(-10,10,( 1,2))    # genera un número decimal entre a y b que será el punto p

print(distance(P0,P1,p))

#72 Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?
Z = np.arange(1,15)

R = np.lib.stride_tricks.as_strided(Z,(11,4),(4,4)) #Divide al arreglo en una nueva matriz de 11x4
print(R)

#77 Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)


#78 Consider a 16x16 array, how to get the block¬sum (block size is 4x4)? 
matriz1= np.ones((16,16))
valor=4
resultado=np.add.reduceat(np.add.reduceat(matriz1,  np.arange(0, matriz1.shape[0], valor), axis=0), 
                         np.arange(0, matriz1.shape[1], valor), axis=1)  # toma los valores de un array y los reduce a un único valor usando una función dada
print(resultado)

#88 Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
print(np.einsum('i->', A)) # np.sum(A)
print(np.einsum('i,i->i', A, B)) # A * B
print(np.einsum('i,i', A, B)) # np.inner(A, B)
print(np.einsum('i,j', A, B)) # np.outer(A, B)
# #La primera representación hace referencia a la suma de los elementos del vector, siendo i todos los elementos de A y con el identificador -> sumamos todos los elementos.
# 	La segunda i toma los valores de A y B y el identificador permite crear una matriz de una dimensión.
# 	El tercero i toma los valores de A y B, sin embargo, como no hay un identificador, internamente nos esta diciendo que sumemos la multiplicación resultante de A * B
# 	El cuarto i y j toman los valores de A y B, en este caso, realiza un producto cruz, y como no hay un identificador, representa los valores como matriz, pero internamente los esta colocando en una matriz resultante de multiplicar cada uno de los valores de A por el arreglo de B.
