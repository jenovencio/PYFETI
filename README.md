<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# PYFETI
PYFETI is a standalone Python Library to solve and implemented parallel FETI-Like solvers using mpi4py.

# Dependecies
* NumPy
* Scipy
* MPI4Py
* Dill
* Pandas
* Matplotlib
* Numdifftools

# Installing PyFETI
Before installing PyFETI we stronly recommend the use of [ANACONDA](https://www.anaconda.com/distribution/) and [git](https://git-scm.com/downloads).
PyFETI is suppose to work in both Windows and Linux system, but is not fully supported, so please let us know with you are facing any problem.

# Linux installation
```{r, engine='bash', count_lines}
mkdir PYFETI
cd PYFETI
git init 
git clone https://username@bitbucket.org/teamsinspace/documentation-tests.git
```

The command above should copy the remote files to your local system. Now, we must activate the Anaconda virtual environment and install PyFETI.

```{r, engine='bash', count_lines}
source activate $you virtual env$
python setup.py install
```

If you install PyFETI, you should run all unittest to make sure everything is properly working.
```{r, engine='bash', count_lines}
cd pyfeti/src/tests
python test_feti_solver.py
```

Almost every source file also contains unittests, so feel free to run all of them.
PyFETI uses mpi4pi and requires the installation of some mpi distriction, see [MSMPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi),
[IntelMPI](https://software.intel.com/en-us/mpi-library), and [OpenMPI](https://www.open-mpi.org/). Because multiple MPI implementation are supported, the user must create
a environment variable to set MPI path that must be used in PyFETI.

```{r, engine='bash', count_lines}
export MPIDIR=/program/mpi
```

Also, you can have multiple python virtual environments, then you must set a environment variable to specify which python.exe to use:


```{r, engine='bash', count_lines}
export 'PYTHON_ENV'=/condaenv/pyfeti
```

Now, it is time to run python and import pyfeti modules.

```{r, engine='bash', count_lines}
python
>>> import pyfeti
```

Have fun!

# Theory behind PyFETI
## Solving with Dual Assembly
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
