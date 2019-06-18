"""
Preconditioning module for PYFETI
"""
from scipy.sparse import csr_matrix, lil_matrix, identity
import numpy as np

all = ['PreconditionerAssembler',
       'NoPreconditioner',
       'LumpedPreconditioner',
       'DirichletPreconditioner']

class PreconditionerAssembler:
    def __init__(self):
        self.preconditioner_operator = csr_matrix([])

    def preallocate(self, lambda_size):
        self.preconditioner_operator = csr_matrix((lambda_size, lambda_size))

    def assemble(self, local_preconditioner):
        self.preconditioner_operator += local_preconditioner

    def precondition(self, search_directions):
        return self.preconditioner_operator.dot(search_directions)

class PreconditionerBase:
    def __init__(self, K_local, B_local):
        self.K_local = K_local
        self.B_local = B_local
        self.Q = None
        self.Q_exp = None
        self.interior_dofs = None
        self.interface_dofs = None

        if not B_local:
            print('WARNING: No interfaces are present. Preconditioners not set.')
        else:
            self._identify_interface_and_interior_dofs()

            self._set_Q()
            self._expand_Q()

    def _identify_interface_and_interior_dofs(self):
        self.interface_dofs = np.array([], dtype=int)
        for interface_id, Bij in self.B_local.items():
            csr_B = csr_matrix(Bij)
            self.interface_dofs = np.unique(np.append(self.interface_dofs, csr_B.indices))

        self.interior_dofs = np.setdiff1d(np.arange(self.K_local.shape[0]), self.interface_dofs)
        self.interior_dofs.astype(dtype=int)

    def _set_Q(self):
        pass

    def _expand_Q(self):
        self.Q_exp = lil_matrix(self.K_local.shape)
        self.Q_exp[np.ix_(self.interface_dofs, self.interface_dofs)] = self.Q

    def build_local_preconditioner(self, B):
        QBT = np.dot(self.Q_exp, B.T)
        return np.dot(B, QBT)

class NoPreconditioner(PreconditionerBase):
    def __init__(self, K_local, B_local):
        super().__init__(K_local, B_local)

    def _set_Q(self):
        self.Q = identity(len(self.interface_dofs))

class LumpedPreconditioner(NoPreconditioner):
    def __init__(self, K_local, B_local):
        super().__init__(K_local, B_local)

    @property
    def K_bb(self):
        if self.interior_dofs is None:
            self._identify_interface_and_interior_dofs()

        return SparseMatrix(self.K_local.data[np.ix_(self.interface_dofs, self.interface_dofs)])

    def _set_Q(self):
        self.Q = self.K_bb.data

class DirichletPreconditioner(LumpedPreconditioner):
    def __init__(self, K_local, B_local):
        super().__init__(K_local, B_local)

    @property
    def K_ii(self):
        if self.interior_dofs is None:
            self._identify_interface_and_interior_dofs()

        return SparseMatrix(self.K_local.data[np.ix_(self.interior_dofs, self.interior_dofs)])

    @property
    def K_ib(self):
        if self.interior_dofs is None:
            self._identify_interface_and_interior_dofs()

        return SparseMatrix(self.K_local.data[np.ix_(self.interior_dofs, self.interface_dofs)])

    def schur_complement(self):
        return cal_schur_complement(self.K_ii, self.K_ib, self.K_bb)

    def _set_Q(self):
        self.Q = self.schur_complement()