from unittest import TestCase
from numpy.testing import assert_array_equal, assert_allclose
from pyfeti.src.preconditioner import *
from pyfeti.src.linalg import SparseMatrix
import numpy as np
from scipy.sparse import csr_matrix

class PreconditionerAssemblerTest(TestCase):
    def setUp(self):
        self.new_local_precond = csr_matrix(np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]]))

        self.precond_assembler_test = PreconditionerAssembler()

    def tearDown(self):
        pass

    def test_preallocate(self):
        self.precond_assembler_test.preallocate(6)
        operator_actual = self.precond_assembler_test.preconditioner_operator

        assert_array_equal(operator_actual.todense(), csr_matrix((6,6)).todense())

    def test_assemble(self):
        self.precond_assembler_test.preallocate(3)
        self.precond_assembler_test.assemble(self.new_local_precond)
        operator_actual = self.precond_assembler_test.preconditioner_operator

        assert_array_equal(operator_actual.todense(), self.new_local_precond.todense())

    def test_precondition(self):
        self.precond_assembler_test.preallocate(3)
        self.precond_assembler_test.assemble(self.new_local_precond)

        vector = np.array([1, 0.5, -1/3])

        assert_allclose(np.array([1, 4.5, 8]), self.precond_assembler_test.precondition(vector))


def create5dofExample():
    K = SparseMatrix(csr_matrix(
        np.array([[2, -3, 0, 0, 0], [-1, 4, -3, 0, 0], [0, -1, 4, -3, 0], [0, 0, -1, 4, -3], [0, 0, 0, -1, 2]])))

    B = {(1, 0): csr_matrix(np.array([[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0]])),
                    (1, 2): csr_matrix(np.array([0, 0, 0, 0, 1]))}

    return K, B

class PreconditionerBaseTest(TestCase):
    def setUp(self):
        self.K_local, self.B_local = create5dofExample()

    def tearDown(self):
        pass

    def test_identify_interface_and_interior_dofs(self):
        preconditioner_test = PreconditionerBase(self.K_local, self.B_local)
        interior_dofs_desired = np.array([2, 3])
        interface_dofs_desired = np.array([0, 1, 4])

        assert_array_equal(preconditioner_test.interior_dofs, interior_dofs_desired)
        assert_array_equal(preconditioner_test.interface_dofs, interface_dofs_desired)

class NoPreconditionerTest(TestCase):
    def setUp(self):
        self.K_local, self.B_local = create5dofExample()
        self.preconditioner_test = NoPreconditioner(self.K_local, self.B_local)

    def tearDown(self):
        pass

    def test_set_Q(self):
        Q_desired = np.identity(3)

        assert_array_equal(self.preconditioner_test.Q.todense(), Q_desired)

    def test_build_local_preconditioner(self):
        local_preconditioner_actual = self.preconditioner_test.build_local_preconditioner(self.B_local[(1, 0)])
        local_preconditioner_desired = np.array([[1.0, 0], [0, 1.0]])

        assert_allclose(local_preconditioner_actual.todense(), local_preconditioner_desired)

        local_preconditioner_actual = self.preconditioner_test.build_local_preconditioner(self.B_local[(1, 2)])
        local_preconditioner_desired = np.array([[1]])

        assert_allclose(local_preconditioner_actual.todense(), local_preconditioner_desired)

class LumpedPreconditionerTest(TestCase):
    def setUp(self):
        self.K_local, self.B_local = create5dofExample()
        self.preconditioner_test = LumpedPreconditioner(self.K_local, self.B_local)

    def tearDown(self):
        pass

    def test_set_Q(self):
        Q_desired = np.array([[2, -3, 0],[-1, 4, 0],[0, 0, 2]])
        assert_array_equal(self.preconditioner_test.Q.todense(), Q_desired)

    def test_build_local_preconditioner(self):
        local_preconditioner_actual = self.preconditioner_test.build_local_preconditioner(self.B_local[(1, 0)])
        local_preconditioner_desired = np.array([[2, -3], [-1, 4]])

        assert_allclose(local_preconditioner_actual.todense(), local_preconditioner_desired)

        local_preconditioner_actual = self.preconditioner_test.build_local_preconditioner(self.B_local[(1, 2)])
        local_preconditioner_desired = np.array([[2]])

        assert_allclose(local_preconditioner_actual.todense(), local_preconditioner_desired)

class DirichletPreconditionerTest(TestCase):
    def setUp(self):
        self.K_local, self.B_local = create5dofExample()
        self.preconditioner_test = DirichletPreconditioner(self.K_local, self.B_local)

    def tearDown(self):
        pass

    def test_set_Q(self):
        Q_desired = np.array([[2, -3, 0],[-1, 3.69230769, -0.69230769],[0, -0.23076923, -0.76923077]])
        assert_allclose(self.preconditioner_test.Q.todense(), Q_desired)

    def test_build_local_preconditioner(self):
        local_preconditioner_actual = self.preconditioner_test.build_local_preconditioner(self.B_local[(1, 0)])
        local_preconditioner_desired = np.array([[2, -3], [-1, 3.69230769]])

        assert_allclose(local_preconditioner_actual.todense(), local_preconditioner_desired)

        local_preconditioner_actual = self.preconditioner_test.build_local_preconditioner(self.B_local[(1, 2)])
        local_preconditioner_desired = np.array([[-0.76923077]])

        assert_allclose(local_preconditioner_actual.todense(), local_preconditioner_desired)


