# needs https://github.com/sagemath/sage/pull/36031

#########################################################################################################
# the code below is an adaptation of the code for mixed volume classification by Christopher Borger
# https://github.com/christopherborger/mixed_volume_classification/blob/master/volume_classification.sage
#########################################################################################################

load("polytopes.sage")

import functools
import itertools
import logging
import os.path
import sys

from collections import defaultdict

from sage.geometry.palp_normal_form import _palp_PM_max, _palp_canonical_order
from sage.geometry.polyhedron.parent import Polyhedra
from sage.misc.lazy_attribute import lazy_attribute

# Using the logging package one can conveniently turn off and on the auxiliary messages

logging.basicConfig(format='%(message)s',stream=sys.stdout,level=logging.INFO)
# After modifying the level from, say, logging.INFO to logging.WARNING , the change will come into force only after _restarting the sage session_ and reloading

# Sandwich is a pair of centrally symmetric lattice polytopes A,B with A being a subset of B.
# For the sake of efficiency, A also comes with its "symmetry-broken" part halfA such that A = halfA \cup -halfA \cup {0}.
# The gap of a sandwich A,B is the difference |B \cap Z^d| - |A \cap Z^d| of the number of integer points in B and A.

# that's the template for names of files, in which we store polytopes
FILE_NAME_DELTA = 'data/dim_%d_delta_%d.txt'
FILE_NAME_DELTA_EXTR = 'data/dim_%d_delta_%d_extremal.txt'


class Sandwich:
    r"""
    A sandwich of lattice polytopes, equipped with a sequence of invariants as keys

    To profile sequences of invariants::

        sage: %prun -s ncalls -l _key_func_ delta_classification(3, 1, False)  # not tested

    EXAMPLES::

        sage: S = Sandwich(Polyhedron([[2, 3], [4, 5], [6, 7]]), Polyhedron([[0, 0], [0, 7], [7, 7], [7, 0]]))
        sage: list(S)

    """
    def __init__(self, A, B, *, A_integral_points=None, B_integral_points=None):
        if isinstance(A, (tuple, list)):
            # Assume it's [halfA, A] where A is a Polyhedron
            self._A = A[1]
            self._halfA = A[0]
        else:
            self._A = A

        if A_integral_points is not None:
            self._A_integral_points = A_integral_points

        if B_integral_points is not None:
            self._B_integral_points = B_integral_points
        elif isinstance(B, (tuple, list)):
            self._B_integral_points = B
        else:
            self._B = B

    def polyhedra_parent(self):
        return self._A.parent()

    def __repr__(self):
        if self.gap():
            return f"Sandwich conv({sorted(self._A.vertices_list())}) ⊆ conv({sorted(self._B.vertices_list())}) with gap {self.gap()}"
        return f"Polytope conv({sorted(self._A.vertices_list())})"

    def plot(self):
        return self._B.plot(alpha=.3, polygon='yellow') + self._A.plot(alpha=.3, polygon='red')

    @lazy_attribute
    def _halfA(self):
        A = self._A
        m = A.ambient_dim()
        return break_symmetry(A, m)

    @lazy_attribute
    def _A_integral_points(self):
        return self._A.integral_points()

    def A_integral_points(self):
        return self._A_integral_points

    def A_integral_points_count(self):
        return len(self.A_integral_points())

    @lazy_attribute
    def _B_integral_points(self):
        return self._B.integral_points()

    @lazy_attribute
    def _B(self):
        return self.polyhedra_parent()([self._B_integral_points, [], []], None)

    def B_integral_points(self):
        r"""
        Return a tuple of immutable vectors
        """
        return self._B_integral_points

    def B_integral_points_count(self):
        return len(self.B_integral_points())

    @cached_method
    def gap(self):
        return self.B_integral_points_count() - self.A_integral_points_count()

    @cached_method
    def _key_func_dimensions(self):
        return (self._A.n_facets(), self._A.n_vertices(), self._B.n_facets(), self._B.n_vertices())

    @lazy_attribute
    def _B_vertex_facet_pairing_matrix(self):
        return self._B.slack_matrix().transpose()

    @cached_method
    def _key_func_B_partitions(self):
        r"""
        Invariants: degree-1 symmetric functions, max symmetric functions of rows and columns
        """
        row_sums = Partition(sorted(sum(self._B_vertex_facet_pairing_matrix.columns()), reverse=True))
        column_sums = Partition(sorted(sum(self._B_vertex_facet_pairing_matrix.rows()), reverse=True))
        column_maxes = Partition(sorted((max(x for x in column)
                                        for column in self._B_vertex_facet_pairing_matrix.columns()),
                                        reverse=True))
        row_maxes = Partition(sorted((max(x for x in row)
                                      for row in self._B_vertex_facet_pairing_matrix.rows()),
                                     reverse=True))

        #print(row_sums, column_sums)
        return row_sums, column_sums, row_maxes, column_maxes

    @lazy_attribute
    def _A_vertex_facet_pairing_matrix(self):
        return self._A.slack_matrix().transpose()

    @staticmethod
    def _row_sums(matrix):
        return Partition(sorted(sum(matrix.columns()), reverse=True))

    @staticmethod
    def _row_power_sums(matrix, powers):
        return tuple(Partition(sorted((sum(x**k for x in row) for row in matrix.rows()),
                                      reverse=True))
                     for k in powers)

    @staticmethod
    def _column_sums(matrix):
        return Partition(sorted(sum(matrix.rows()), reverse=True))

    @staticmethod
    def _column_power_sums(matrix, powers):
        return tuple(Partition(sorted((sum(x**k for x in column) for column in matrix.columns()),
                                      reverse=True))
                     for k in powers)

    @staticmethod
    def _row_maxes(matrix):
        return Partition(sorted((max(x for x in row) for row in matrix.rows()),
                                reverse=True))

    @staticmethod
    def _column_maxes(matrix):
        return Partition(sorted((max(x for x in column) for column in matrix.columns()),
                                reverse=True))

    @staticmethod
    def _row_and_column_sums_and_maxes(matrix):
        return (Sandwich._row_sums(matrix), Sandwich._column_sums(matrix),
                Sandwich._row_maxes(matrix), Sandwich._column_maxes(matrix))

    @staticmethod
    def _row_and_column_power_sums(matrix, powers):
        return (Sandwich._row_power_sums(matrix, powers), Sandwich._column_power_sums(matrix, powers))

    @cached_method
    def _key_func_A_partitions(self):
        r"""
        Invariants: degree-1 symmetric functions, max symmetric functions of rows and columns
        """
        return Sandwich._row_and_column_sums_and_maxes(self._A_vertex_facet_pairing_matrix)

    @lazy_attribute
    def _A_vertex_B_facet_pairing_matrix(self):

        Vrep_matrix = matrix(ZZ, self._A.Vrepresentation())
        Hrep_matrix = matrix(ZZ, self._B.Hrepresentation())

        # Getting homogeneous coordinates of the Vrepresentation.
        hom_helper = matrix(ZZ, [1 if v.is_vertex() else 0 for v in self._A.Vrepresentation()])
        hom_Vrep = hom_helper.stack(Vrep_matrix.transpose())

        PM = Hrep_matrix * hom_Vrep
        PM.set_immutable()
        return PM

    @cached_method
    def _key_func_A_vertex_B_facet_partitions(self):
        #return Sandwich._row_and_column_sums_and_maxes(self._A_vertex_B_facet_pairing_matrix)
        return Sandwich._row_and_column_power_sums(self._A_vertex_B_facet_pairing_matrix, range(1, 3))

    @cached_method(do_pickle=True)
    def _key_func_B_permutation_normal_form(self):
        PNF = self._B_vertex_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        PNF.set_immutable()
        return PNF

    @cached_method(do_pickle=True)
    def _key_func_A_vertex_B_facet_permutation_normal_form(self):
        PNF = self._A_vertex_B_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        PNF.set_immutable()
        return PNF

    @lazy_attribute
    def _LLP(self):
        return layered_polytope_from_sandwich((None, self._A), self._B)

    @lazy_attribute
    def _LLP_vertex_facet_pairing_matrix(self):
        return self._LLP.slack_matrix().transpose()

    @lazy_attribute
    def _LLP_PM_max_and_permutations(self):
        PM_max, permutations = _palp_PM_max(self._LLP_vertex_facet_pairing_matrix, check=True)
        PM_max.set_immutable()
        return PM_max, permutations

    @cached_method(do_pickle=True)
    def _key_func_LLP_permutation_normal_form(self):
        "FIXME: This is apparently NOT a normal form"
        #return self._LLP.normal_form(algorithm='palp')  # fastest of all, but crashes for dim > 3.
        #PNF = self._LLP_vertex_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        #PNF = self._LLP._palp_PM_max(check=False)       # slower (before https://github.com/sagemath/sage/pull/35997), much faster (after)

        #PNF.set_immutable()
        PNF = self._LLP_PM_max_and_permutations[0]       # same as above, but stores permutations for use by _key_func_LLP_palp_native_normal_form below
        return PNF

    @cached_method(do_pickle=True)
    def _key_func_A_palp_native_normal_form(self):
        return tuple(self._A.normal_form())

    @cached_method(do_pickle=True)
    def _key_func_LLP_palp_native_normal_form(self):
        PM_max, permutations = self._LLP_PM_max_and_permutations
        return tuple(_palp_canonical_order(self._LLP.vertices(), PM_max, permutations)[0])

    def key_funcs(self):
        if self.gap():
            return (self._key_func_dimensions,
                    self._key_func_A_vertex_B_facet_partitions)
        return (self._key_func_dimensions,
                #self._key_func_A_partitions,
                #self._key_func_B_partitions,
                self._key_func_A_vertex_B_facet_partitions,
                #self._key_func_B_permutation_normal_form,
                #self._key_func_A_vertex_B_facet_permutation_normal_form,
                #self._key_func_LLP_permutation_normal_form,
                #self._key_func_LLP_palp_native_normal_form,   # Actually no need to construct LLP b/c gap=0
                self._key_func_A_palp_native_normal_form)

    @staticmethod
    def key_costs():
        return (0,
                1,
                50)

    def item_cost(self, i):
        try:
            if self.key_funcs()[i].is_in_cache():
                return 0
        except AttributeError:
            pass
        return self.key_costs()[i]

    def __len__(self):
        return len(self.key_funcs())

    def __getitem__(self, i):
        r"""
        Return the components of the key for the trie
        """
        return self.key_funcs()[i]()

    @cached_method
    def noninvariant_keys(self):
        return (self._halfA,
                tuple(sorted(tuple(int(x) for x in v) for v in self._B.vertices())))

    def __eq_noninvariant__(self, other):
        if self is other:
            return True
        return self.noninvariant_keys() == other.noninvariant_keys()

    def __eq__(self, other):
        # First check fast non-invariant
        if self.__eq_noninvariant__(other):
            return True
        return all(s == o for s, o in zip(self, other))


def dict_factory(key_prefix):
    return dict()


def make_diskcache_Index_factory(dirname):
    try:
        import diskcache
    except ImportError:
        raise ImportError('Use !pip install diskcache')
    def Index_factory(key_prefix):
        return diskcache.Index(dirname + f'_{len(key_prefix)}')
    return Index_factory


class SandwichStorage:
    r"""
    Minimal implementation of a dictionary with hierarchical lazy keys.

    Strictly worse than a proper lazy trie because everything is stashed into large dictionaries.

    INPUT:

    - ``mapping_factory`` -- Constructor for a :class:`dict` or other mapping

    EXAMPLES::

        sage: d = SandwichStorage()
        sage: d['aaaa'] = 1
        sage: d._mapping_list
        sage: d['aaba'] = 2
        sage: d._mapping_list
        sage: d['aabb'] = 3
        sage: d._mapping_list
        sage: d['aaaa'] = 7  # FIXME: overwriting creates a long chain
        sage: d._mapping_list

    Persistent sandwich storage using diskcache::

        sage: d = SandwichStorage(mapping_factory=make_diskcache_Index_factory('doctest_SandwichStorage'))
        sage: d['aaaa'] = 1
        sage: d['aaba'] = 2
        sage: d['aabb'] = 3
        sage: d['aaaa'] = 7
        sage: del d

    Later, perhaps in another process::

        sage: dd = SandwichStorage(mapping_factory=make_diskcache_Index_factory('doctest_SandwichStorage'))
        sage: dd['aabb']
        3
        sage: sorted(dd.values())
        [2, 3, 7]
    """
    def __init__(self, mapping_factory=None):
        if mapping_factory is None:
            mapping_factory = dict_factory
        self._mapping_factory = mapping_factory
        self._mapping_list = [mapping_factory(())]  # key_length -> key_prefix -> (key, value) | 'not_unique'

    def _key_prefix_to_mapping(self, key_prefix):
        length = len(key_prefix)
        while length >= len(self._mapping_list):
            self._mapping_list.append(self._mapping_factory(key_prefix[:len(self._mapping_list)]))
        return self._mapping_list[length]

    def _sufficient_key_prefix(self, key):
        r"""
        Return the shortest prefix of ``key`` that suffices to either:
        - identify a unique candidate for ``key``, in which case
          ``(key_prefix, (candidate_key, candidate_value), checked)`` is returned;
          when ``checked`` is True, the ``candidate_key`` is already a known hit.
        - show that ``key`` is not in ``self``, in which case ``(key_prefix, None, False)`` is returned

        OUTPUT: a tuple
        """
        key_prefix = ()
        while True:
            mapping = self._key_prefix_to_mapping(key_prefix)
            try:
                item = mapping[key_prefix]
            except KeyError:
                return key_prefix, None, False
            else:
                if item != 'not_unique':
                    return key_prefix, item, False

            # possible improvement: insisting on a unique candidate is too much when the next key element
            # is too expensive. When the subtrie has <= THRESHOLD candidates, it may be faster to invert:
            # loop through all candidates and do the fast non-invariant check.
            if (cost := self._key_cost(key, len(key_prefix))) > 1:
                next_mapping = self._key_prefix_to_mapping(key_prefix + (None,))
                if not isinstance(next_mapping, dict) or len(next_mapping) <= cost:  # len is expensive for diskcache.Index
                    for i, (same_prefix, item) in enumerate(next_mapping.items()):
                        if item != 'not_unique':
                            if item[0].__eq_noninvariant__(key):
                                return key_prefix + (self._key_item(item, 0),), item, True
                        if i >= cost:
                            break
            try:
                key_prefix = key_prefix + (self._key_item(key, len(key_prefix)),)
            except IndexError:
                assert False, 'path cannot end with not_unique'

    def _key_item(self, key, index):
        return key[index]

    def _key_cost(self, key, index):
        return key.item_cost(index)

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __getitem__(self, key):
        key_prefix, item, checked = self._sufficient_key_prefix(key)
        if item is None:
            raise KeyError(key)
        candidate_key, candidate_value = item
        if len(key_prefix) == len(key) or checked:
            return candidate_value
        if candidate_key.__eq_noninvariant__(key):
            return candidate_value
        for i in range(len(key_prefix), len(key)):
            if self._key_item(candidate_key, i) != self._key_item(key, i):
                raise KeyError(key)  # f'{key!r}; _sufficient_key_prefix returned {key_prefix=} {item=}')
        return candidate_value

    def __setitem__(self, key, value):
        key_prefix, item, checked = self._sufficient_key_prefix(key)
        mapping = self._key_prefix_to_mapping(key_prefix)
        if checked or item is not None:
            candidate_key, candidate_value = item
            while len(key_prefix) < len(key):
                candidate_next = candidate_key[len(key_prefix)]
                key_next = key[len(key_prefix)]

                candidate_key_prefix = key_prefix + (candidate_next,)
                candidate_mapping = self._key_prefix_to_mapping(candidate_key_prefix)
                candidate_mapping[candidate_key_prefix] = item

                mapping[key_prefix] = 'not_unique'  # atomic

                key_prefix = key_prefix + (key_next,)
                mapping = self._key_prefix_to_mapping(key_prefix)
                if candidate_next != key_next:
                    break
        mapping[key_prefix] = (key, value)

    def keys(self):
        for key, value in self.items():
            yield key

    __iter__ = keys

    def __len__(self):
        return len(list(iter(self)))  # FIXME obviously

    def values(self):
        for key, value in self.items():
            yield value

    def items(self):
        key_prefix = []  # currently only the length matters
        not_unique = True
        while not_unique:
            not_unique = False
            mapping = self._key_prefix_to_mapping(key_prefix)
            for item in mapping.values():
                if item == 'not_unique':
                    not_unique = True
                else:
                    yield item
            if not_unique:
                key_prefix.append(None)


class SandwichStorage_with_diskcache_Cache(SandwichStorage):
    r"""
    Uses a :class:`diskcache.Cache` to store a mapping: noninvariant_keys -> Sandwich.
    """
    def __init__(self, mapping_factory=None, cache=None):
        super().__init__(mapping_factory=mapping_factory)
        self._cache = cache

    def _key_item(self, key, index):
        try:
            cost = key.item_cost(index)
        except AttributeError:
            pass
        else:
            if cost >= 42:
                try:
                    cached = self._cache[key.noninvariant_keys()]
                except KeyError:
                    pass
                    #print(f"Miss: {key.noninvariant_keys()}")
                else:
                    #print(f"Hit: {key.noninvariant_keys()}, {cached.item_cost(index)}")
                    if not cached.item_cost(index):
                        result = cached[index]
                        key.key_funcs()[index].set_cache(result)
                        return result
                result = key[index]
                key.key_funcs()[index].set_cache(result)
                #print(f"Store: {key.noninvariant_keys()}")
                self._cache[key.noninvariant_keys()] = key
                return result

        return key[index]


def break_symmetry(A,m):
    """
    	takes a centrally symmetric m-dimensional polytope A
    	computes a subset halfA of its vertices I such that I = conv(halfA \cup -halfA)
    """
    halfA = []
    for z in A.vertices():
        if next((x for x in z if x != 0), None) > 0:
            l = tuple(int(x) for x in z)
            halfA.append(l)
    return tuple(sorted(halfA))


def do_not_break_symmetry(A, m):
    return tuple(sorted(tuple(int(x) for x in z) for z in A.vertices()))


def is_extendable(S,v,Delta):
    """
        Check whether the extension of a set S of vectors by a vector v causes a determinant to exceed Delta.
    """
    m = len(v)
    for C in Combinations(S,m-1):
        M = matrix(C + [list(v)])
        if abs(det(M)) > Delta:
            return false    
    return true


def layered_polytope_from_sandwich(A,B):
    """ 3*B is embedded into height 0, two copies of 3*A are embedded into heights 1 and -1.
        Then, one generates a polytope based on these three layers at heights -1,0 and 1
        Note: If A and B are centrally symmetric, then the resulting polytope is centrally symmetric as well.
    """
    middleLayer = [tuple(3*vector(v))+(0,) for v in B.vertices()]
    upperLayer = [tuple(3*vector(v))+(1,) for v in A[1].vertices()]
    lowerLayer = [tuple(3*vector(v))+(-1,) for v in A[1].vertices()]
    return Polyhedron(middleLayer+upperLayer+lowerLayer, backend=B.backend())


# Sandwich factory is used to store sandwiches up to affine unimodular transformations.
# A sandwich factory is a dictionary of dictionaries. For each possible gap, a storage
# for sandwiches with this gap is created. The latter storage
# is a dictionary with key,value pairs such that the value is a sandwich and
# the respective key is the sandwich normal form of this sandwich.


sandwich_hits = 0
sandwich_failures = 0


class SandwichFactory(defaultdict):

    def __init__(self, m, Delta, mode, polyhedra_backend='ppl'):
        super().__init__(SandwichStorage)
        self._m = m
        self._Delta = Delta

        # Normalize computation mode
        if not mode:
            mode = 'delta'
        elif mode is True:
            mode = 'delta_ext'

        if mode not in ['delta', 'delta_ext', 'delta_cone']:
            raise ValueError("Unknown computation mode", mode)

        self._mode = mode
        if mode == 'delta_ext':
            # set the known lower bound for h(Delta,m) by Lee et al.
            self._cmax = m^2 - m + 1 *2*m*Delta
        self._deque = []

        self._polyhedra_backend = polyhedra_backend
        self._polyhedra_parent = Polyhedra(ZZ, m, backend=polyhedra_backend)

    def prepare_sandwiches(self):
        m = self._m
        Delta = self._Delta
        mode = self._mode

        if Delta == 2:
            HNFs = []
            for nonzeros in range(m):
                R = matrix.identity(m)
                for i in range(m-nonzeros-1,m):
                    R[i, m-1] += 1
                HNFs.append(R)
        else:
            HNFs = delta_normal_forms(m,Delta)

        for basisA in HNFs:
            # first, we generate A and halfA out of basisA
            mbA = matrix(basisA)
            if mode == 'delta_cone':
                # Start with positive HNF vectors only; this breaks symmetry
                A_points = mbA.augment(vector(ZZ, m))
                mA = mbA.augment(-mbA)
            else:
                mA = mbA.augment(-mbA)
                A_points = mA
            A = self._polyhedra_parent([A_points.transpose(), [], []], None, convert=True)

            if mode == 'delta_cone':  # FIXME: Should probably get rid of halfA altogether
                halfA = do_not_break_symmetry(A, m)
            else:
                halfA = break_symmetry(A,m)

            # second, the outer container B is the centrally symmetric parallelotope spanned by the vectors in basisA
            B = polytopes.parallelotope(mA.transpose(), backend=self._polyhedra_backend)

            # B may contain some integral points that are Delta-too-large with respect to A, and so we do:
            sandwich = Sandwich([halfA,A], B)
            yield self.reduce_sandwich([halfA,A], sandwich)

    def reduce_sandwich(self, newA, sandwich):
        """
        For a given sandwich (A,B) and a value of Delta
        the function returns a polytope
        obtained by removing all of the lattice points v of B
        with the property that if v is added to A, there will be a determinant of absolute value > Delta
        """
        Delta = self._Delta
        mode = self._mode

        to_be_removed = set()
        to_be_kept = set()

        Z = sandwich.B_integral_points()
        for v in Z:
            if v in newA[1]:
                continue
            if v in to_be_removed or v in to_be_kept:  ## this just avoids considering -w in case that w was considered already before
                continue
            mv = -v
            mv.set_immutable()
            if mode == 'delta_cone':
                if mv/1000 in newA[1]:
                    # never extend to a non-pointed cone
                    to_be_removed.add(v)
                    continue
            if is_extendable(newA[0],v,Delta):
                to_be_kept.add(v)
                if mode != 'delta_cone':
                    to_be_kept.add(mv)
            else:
                to_be_removed.add(v)
                if mode != 'delta_cone':
                    to_be_removed.add(mv)
        if to_be_removed:
            newB = tuple(z for z in Z if z not in to_be_removed)
            return Sandwich(newA, newB)
        else:
            return Sandwich(newA, sandwich._B, B_integral_points=Z)

    def append_sandwich(self, sandwich):
        """
            If no affine unimodular image of the sandwich (A,B) is in the sandwich factory self,
            the sandwich (A,B) is appended to self.
        """
        global sandwich_hits, sandwich_failures

        Gap = sandwich.gap()

        # crucial that sandwich is a LatticePolytope (or something else with a good hash),
        # not a Polyhedron (which has a poor hash)
        if sandwich not in self[Gap]:
            self[Gap][sandwich] = [(sandwich._halfA, sandwich._A), sandwich._B]
            if not Gap:
                print(sandwich)
            sandwich_failures += 1
            return sandwich
        else:
            sandwich_hits += 1
            return None

    def __repr__(self):
        return f'{self.__class__.__name__} with keys {sorted(self)}'

    def branch_sandwich(self, sandwich):

        A = (sandwich._halfA, sandwich._A)
        B = sandwich._B

        for v in B.vertices(): # pick a vertex of B which is not in A

            if self._mode == 'delta_cone':
                if not v:
                    continue

            if v not in A[1]:
                break

        v = vector(v, immutable=True)
        mv = -v

        if self._mode == 'delta_cone':
            points_added = [v]
        else:
            points_added = [v, mv]

        blow_up_of_A = self._polyhedra_parent([list(A[1].vertices()) + points_added, [], []],
                                              None,
                                              convert=True)  ## this uses that all points in B are "Delta-ok" for A
        if self._mode == 'delta_cone':
            half_of_blow_up_of_A = do_not_break_symmetry(blow_up_of_A, self._m)
        else:
            half_of_blow_up_of_A = break_symmetry(blow_up_of_A, self._m)
        newA = [half_of_blow_up_of_A, blow_up_of_A]
        sandwich1 = self.reduce_sandwich(newA, sandwich)

        reduction_of_B = tuple(z for z in sandwich.B_integral_points()
                               if z not in points_added)
        sandwich2 = Sandwich(A, reduction_of_B, A_integral_points=sandwich.A_integral_points())
        if self._mode == 'delta_ext':
            if sandwich1.B_integral_points_count() >= self._cmax:
                yield sandwich1
                npts_blow_up = sandwich1.A_integral_points_count()
                if npts_blow_up > self._cmax:
                    self._cmax = npts_blow_up
            if sandwich2.B_integral_points_count() >= self._cmax:
                yield sandwich2
        else:
            yield sandwich1
            yield sandwich2


class SandwichFactory_with_diskcache_Index(SandwichFactory):
    r"""
    gap -> SandwichStorage

    On macOS, use 'ulimit -n 2048' before starting Sage to avoid running into 'Too many open files'
    """
    def __init__(self, m, Delta, mode, dirname, **kwds):
        super().__init__(m, Delta, mode, **kwds)

        try:
            import diskcache
        except ImportError:
            raise ImportError('Use !pip install diskcache')

        self._dirname = dirname
        self._sandwich_cache = diskcache.Cache(self._dirname + f'_invariants')

        self._deque = diskcache.Deque(directory=dirname + '_deque')
        print(f'Loaded deque of length {len(self._deque)}')

    def __missing__(self, key):
        mapping_factory = make_diskcache_Index_factory(self._dirname + f'_gap{key}')
        #mapping_factory = None  # we are testing only the Cache now
        value = SandwichStorage_with_diskcache_Cache(mapping_factory, cache=self._sandwich_cache)
        print(f"Creating SandwichStorage_with_diskcache_Cache for gap {key}")  # ; size = {len(value)}")  -- expensive to take len()
        self[key] = value
        return value

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dirname!r}) with keys {sorted(self)}'


def new_sandwich_factory(m, Delta, mode, dirname=None, **kwds):

    # Using https://github.com/mina86/pygtrie (https://pygtrie.readthedocs.io/en/latest/#pygtrie.Trie)
    # seemed promising, but unfortunately it always eagerly uses the whole key
    # when creating a new node (in _set_node).
    # (Our SandwichStorage does that only when we overwrite an item, which
    # we never do here.)
    #from pygtrie import Trie
    #sandwich_factory = defaultdict(Trie)

    if dirname is None:
        sandwich_factory = SandwichFactory(m, Delta, mode, **kwds)
    else:
        dirname += f'_m{m}_Delta{Delta}'
        if mode in ['delta_ext', True]:
            dirname += '_ext'
        elif mode == 'delta_cone':
            dirname += '_cone'
        sandwich_factory = SandwichFactory_with_diskcache_Index(m, Delta, mode, dirname, **kwds)

    return sandwich_factory


def sandwich_factory_statistics(sf):
    logging.info("Maximum gap in sandwiches: %d",max(sf.keys()))
    logging.info("Number of sandwiches: %d",sum([len(sf[Gap]) for Gap in sf.keys() if Gap!=0]))
    if 0 in sf.keys():
        logging.info("Number of polytopes found: %d", len(sf[0]))
    logging.info(f"Sandwich normal form hits: {sandwich_hits}, failures: {sandwich_failures}")
    logging.info(50*"-")


def delta_classification(m, Delta, mode, dirname=None, *, order='gap', iterations=None,
                         polyhedra_backend='ppl'):
    """
    Run the sandwich factory algorithm.

    INPUT:

    - ``mode`` -- one of

      - ``'delta'`` -- classify all centrally symmetric m-dimensional lattice polytopes
        with largest determinant equal to Delta

      - ``'delta_ext'`` -- only include the extremal examples attaining h(Delta,m)

      - ``'delta_cone' -- oriented, non--centrally symmetric version
    """
    sf = new_sandwich_factory(m, Delta, mode, dirname=dirname,
                              polyhedra_backend=polyhedra_backend)

    match order:
        case 'gap':
            for sandwich in sf.prepare_sandwiches():
                sf.append_sandwich(sandwich)
            maxGap = max(sf.keys())
            while maxGap > 0:
                sandwich_factory_statistics(sf)
                for sandwich in sf[maxGap]:
                    for new_sandwich in sf.branch_sandwich(sandwich):
                        sf.append_sandwich(new_sandwich)
                del sf[maxGap]
                maxGap = max(sf.keys())
            sandwich_factory_statistics(sf)

        case _:
            deque = sf._deque
            for sandwich in sf.prepare_sandwiches():
                if sf.append_sandwich(sandwich) is not None:
                    deque.append(sandwich)

            iteration = 0

            while deque:
                iteration += 1
                match order:
                    case 'dfs':
                        sandwich = deque.pop()
                    case 'bfs':
                        sandwich = deque.popleft()
                    case 'random':
                        with deque.transact():
                            index = randint(0, len(deque) - 1)
                            sandwich = deque[index]
                            deque[index] = None
                    case _:
                        raise ValueError(f'unknown order parameter: {order}')
                if sandwich is None:
                    continue
                for new_sandwich in sf.branch_sandwich(sandwich):
                    if sf.append_sandwich(new_sandwich) is not None:
                        if new_sandwich.gap():
                            deque.append(new_sandwich)

                if iteration % 2000 == 0:
                    sandwich_factory_statistics(sf)  # very expensive when using diskcache.Deque

                if iterations is not None and iteration > iterations:
                    break

    result = []
    for A,B in sf[0].values():
        result.append(A[1])  ## only store the polytope in A

    return result


def plot_delta_classification(m, Delta=None, mode=None, L=None):
    return graphics_array([P.plot(xmin=-Delta, xmax=Delta,
                                  ymin=-Delta, ymax=Delta,
                                  axes=True, ticks=[[], []],
                                  gridlines=[range(-Delta,Delta+1),
                                             range(-Delta,Delta+1)])
                           for P in L],
                          ncols=6)


## Code below uses boolean "extremal"; above has been generalized to "mode"


def update_delta_classification_database(m,Delta,extremal):
    # the files storing polytopes are created in the data subfolder
    if not os.path.exists('data'):
        os.mkdir('data')

    # let's see whether the file for the pair (m,Delta) is missing
    if (extremal):
        missingDelta = not os.path.isfile(FILE_NAME_DELTA_EXTR % (m,Delta))
    else:
        missingDelta = not os.path.isfile(FILE_NAME_DELTA % (m,Delta))

    if missingDelta:
        # we should run the delta classification

        if (extremal):
            f = open(FILE_NAME_DELTA_EXTR % (m,Delta),'w')
            if (os.path.isfile(FILE_NAME_DELTA % (m,Delta))):
                g = open(FILE_NAME_DELTA % (m,Delta),'r')
                L = eval(g.read().replace('\n',' '))
                g.close()
                hdm = generalized_heller_constant(m,Delta,false)[0]
                result = []
                for P in L:
                    if (Polyhedron(P).integral_points_count() == hdm):
                        result.append(P)
                print([P for P in result],file=f)
                f.close()
            else:
                result = delta_classification(m,Delta,extremal)
                print([[tuple(p) for p in P.vertices()] for P in result],file=f)
                f.close()
        else:
            result = delta_classification(m,Delta,extremal)
            f = open(FILE_NAME_DELTA % (m,Delta),'w')
            print([[tuple(p) for p in P.vertices()] for P in result],file=f)
            f.close()


def lattice_polytopes_with_given_dimension_and_delta(m,Delta,extremal):
    """
        That's the main function for users of this module. It returns the list of all [extremal=false] or only h(Delta,m)-attaining [extremal=true]
        m-dimensional centrally symmetric lattice polytopes with delta equal to Delta.
    """
    # first, we update the database of lattice polytopes with a given delta
    update_delta_classification_database(m,Delta,extremal)

    # now, we can read the list of polytopes from the corresponding file and return them
    if (extremal):
        f = open(FILE_NAME_DELTA_EXTR % (m,Delta),'r')
    else:
        f = open(FILE_NAME_DELTA % (m,Delta),'r')

    L = eval(f.read().replace('\n',' '))
    f.close()
    return [Polyhedron(P) for P in L]


def generalized_heller_constant(m,Delta,extremal):
    """
        Compute the generalized Heller constant h(Delta,m) and a point set attaining it
    """

    DeltaPolytopes = lattice_polytopes_with_given_dimension_and_delta(m,Delta,extremal)
    nmax = 0
    for P in DeltaPolytopes:
        npoints = P.integral_points_count()
        if npoints > nmax:
            nmax = npoints
            Pmax = P
    return nmax , Pmax, len(DeltaPolytopes)

