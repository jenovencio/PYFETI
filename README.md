# PYFETI
PYFETI is a standalone Python Library to solve and implemented parallel FETI-Like solvers using mpi4py.

#Dependecies
* NumPy
* Scipy
* MPI4Py
* Dill
* Pandas


#Solving with Dual Assembly
The PyFETI library is intend to provide easy function in order to solve, the dual assembly problem, namely:


$$
\begin{bmatrix} K & B^{T} \\
                 B & 0  
\end{bmatrix}
\begin{bmatrix} u \\ 
\lambda \end{bmatrix}
=
\begin{bmatrix} f \\ 
0 \end{bmatrix}
$$

Generally the block matrix $K$ is singular due to local rigid body modes, then the inner problem is regularized by adding a subset of the inter-subdomain compatibility requeriments:


$$
\begin{bmatrix} K & B^TG^{T} & B^{T} \\
                GB & 0 & 0   \\
                B & 0 & 0   \\
\end{bmatrix}
\begin{bmatrix} u \\ 
\alpha \\
\lambda \end{bmatrix}
=
\begin{bmatrix} f \\ 
0 \\
0 \end{bmatrix}
$$

Where $G$ is defined as $-R^TB^T$.

The Dual Assembly system of equation describe above can be broken in two equations.

\begin{equation}
Ku + B^{T}\lambda  = f \\
Bu = 0 
\end{equation}

Then, the solution u can be calculate by:

\begin{equation}
u =  K^*(B^{T}\lambda  + f) +  R\alpha \\
\end{equation}

Where $K^*$ is the generelize pseudo inverse and $R$ is $Null(K) = \{r \in R: Kr=0\}$, named the kernel of the K matrix.
In order to the solve $u$ the summation of all forces in the subdomain, interface, internal and extenal forces must be in the image of K. This implies the $(B^{T}\lambda  + f)$ must be orthonal to the null space of K.

\begin{equation}
R(B^{T}\lambda  + f) = 0 \\
\end{equation}

Phisically, the equation above enforces the self-equilibrium for each sub-domain. Using the compatibility equation and the self-equilibrium equation, we can write the dual interface equilibrium equation as:


$$
\begin{bmatrix} F & G^{T} \\
                 G & 0  
\end{bmatrix}
\begin{bmatrix} \lambda  \\ 
\alpha
\end{bmatrix}
=
\begin{bmatrix} d \\ 
e \end{bmatrix}
$$

Where $F = BK^*B^T$, $G = -R^TB^T$, $d = BK^*f$ and $e =- R^Tf $.



# ToDo List
* Unittest with more than 10 Domains
* Easy access to the Parallel F operator
* Parallel Coarse Grip Problem
