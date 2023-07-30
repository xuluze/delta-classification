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

from sage.geometry.lattice_polytope import LatticePolytope, _palp_canonical_order
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
    def __init__(self, A, B):
        if isinstance(A, (tuple, list)):
            # Assume it's [halfA, A] where A is a Polyhedron
            A = A[1]
        self._A = A
        self._B = B

    @cached_method
    def gap(self):
        return self._B.integral_points_count() - self._A.integral_points_count()

    @cached_method
    def _key_func_dimensions(self):
        return (self._A.n_facets(), self._A.n_vertices(), self._B.n_facets(), self._B.n_vertices())

    @lazy_attribute
    def _A_LP(self):
        return LatticePolytope(self._A.vertices_list())

    @lazy_attribute
    def _B_LP(self):
        return LatticePolytope(self._B.vertices_list())

    @lazy_attribute
    def _B_vertex_facet_pairing_matrix(self):
        return self._B_LP.vertex_facet_pairing_matrix()

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
        return self._A_LP.vertex_facet_pairing_matrix()

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
        V = self._A_LP.vertices()
        nv = self._A_LP.nvertices()
        PM = matrix(ZZ, [n * V + vector(ZZ, [c] * nv)
                         for n, c in zip(self._B_LP.facet_normals(), self._B_LP.facet_constants())])
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
        return layered_lattice_polytope_from_sandwich((None, self._A), self._B)

    @lazy_attribute
    def _LLP_vertex_facet_pairing_matrix(self):
        return self._LLP.vertex_facet_pairing_matrix()

    @lazy_attribute
    def _LLP_PM_max_and_permutations(self):
        PM_max, permutations = self._LLP._palp_PM_max(check=True)
        PM_max.set_immutable()
        return PM_max, permutations

    @cached_method(do_pickle=True)
    def _key_func_LLP_permutation_normal_form(self):
        "FIXME: This is apparently NOT a normal form"
        #return self._LLP.normal_form(algorithm='palp')  # fastest of all, but crashes for dim > 3.
        #PNF = self._LLP_vertex_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        #PNF = self._LLP._palp_PM_max(check=False)       # slower (before https://github.com/sagemath/sage/pull/35997), much faster (after)
        
        #PNF.set_immutable()
        PNF = self._LLP_PM_max_and_permutations[0]
        return PNF

    @cached_method(do_pickle=True)
    def _key_func_LLP_palp_native_normal_form(self):
        #breakpoint()
        #return self._LLP.normal_form(algorithm='palp_native')
        PM_max, permutations = self._LLP_PM_max_and_permutations
        return _palp_canonical_order(self._LLP.vertices(), PM_max, permutations)[0]

    def key_funcs(self):
        return (self._key_func_dimensions,
                #self._key_func_A_partitions,
                #self._key_func_B_partitions,
                self._key_func_A_vertex_B_facet_partitions,
                #self._key_func_B_permutation_normal_form,
                #self._key_func_A_vertex_B_facet_permutation_normal_form,
                #self._key_func_LLP_permutation_normal_form,
                self._key_func_LLP_palp_native_normal_form)

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
        return (tuple(sorted(tuple(int(x) for x in v) for v in self._A.vertices())),
                tuple(sorted(tuple(int(x) for x in v) for v in self._B.vertices())))

    def __eq_noninvariant__(self, other):
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
                if len(next_mapping := self._key_prefix_to_mapping(key_prefix + (None,))) <= cost:
                    for same_prefix, item in next_mapping.items():
                        if item != 'not_unique':
                            if item[0].__eq_noninvariant__(key):
                                return key_prefix + (self._key_item(item, 0),), item, True

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
        cost = key.item_cost(index)
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


def prepare_sandwiches(m,Delta):
    for basisA in delta_normal_forms(m,Delta):
        # first, we generate A and halfA out of basisA
        mbA = matrix(basisA)
        mA = mbA.augment(-mbA)
        A = Polyhedron(mA.transpose())
        halfA = break_symmetry(A,m)
    
        # second, the outer container B is the centrally symmetric parallelotope spanned by the vectors in basisA
        B = polytopes.parallelotope(mA.transpose())
        
        # B may contain some integral points that are Delta-too-large with respect to A, and so we do:
        B = reduce_sandwich([halfA,A],B,Delta)
        yield [halfA,A],B


def break_symmetry(A,m):
    """
    	takes a centrally symmetric m-dimensional polytope A
    	computes a subset halfA of its vertices I such that I = conv(halfA \cup -halfA)
    """
    vertA = [vector(z) for z in A.vertices_list()]
    halfA = []
    for l in vertA:
    	if (-l in halfA):
    		continue
    	halfA.append(l)
    return halfA


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


def reduce_sandwich(A,B,Delta):
    """
        For a given sandwich (A,B) and a value of Delta
        the function returns a polytope
        obtained by removing all of the lattice points v of B 
        with the property that if v is added to A, there will be a determinant of absolute value > Delta
    """
    to_be_removed = []
    to_be_kept = []
    for v in B.integral_points():
    	if v in A[1]:
    		continue
    	if (v in to_be_removed or v in to_be_kept):	## this just avoids considering -w in case that w was considered already before
    		continue
    	if is_extendable(A[0],v,Delta):
    		to_be_kept.append(vector(v))
    		to_be_kept.append(-vector(v))
    	else:
    		to_be_removed.append(vector(v))
    		to_be_removed.append(-vector(v))
    Z = [vector(z) for z in B.integral_points()]
    return Polyhedron([z for z in Z if z not in to_be_removed])


def layered_lattice_polytope_from_sandwich(A,B):
    """ 3*B is embedded into height 0, two copies of 3*A are embedded into heights 1 and -1.
        Then, one generates a polytope based on these three layers at heights -1,0 and 1
        Note: If A and B are centrally symmetric, then the resulting polytope is centrally symmetric as well.
    """ 
    middleLayer = [tuple(3*vector(v))+(0,) for v in B.vertices()]
    upperLayer = [tuple(3*vector(v))+(1,) for v in A[1].vertices()]
    lowerLayer = [tuple(3*vector(v))+(-1,) for v in A[1].vertices()]
    return LatticePolytope(middleLayer+upperLayer+lowerLayer)


# Sandwich factory is used to store sandwiches up to affine unimodular transformations.
# A sandwich factory is a dictionary of dictionaries. For each possible gap, a storage
# for sandwiches with this gap is created. The latter storage
# is a dictionary with key,value pairs such that the value is a sandwich and 
# the respective key is the sandwich normal form of this sandwich.


sandwich_hits = 0
sandwich_failures = 0


class SandwichFactory(defaultdict):

    def __init__(self):
        super().__init__(SandwichStorage)

    def append_sandwich(self, A, B):
        """
            If no affine unimodular image of the sandwich (A,B) is in the sandwich factory self,
            the sandwich (A,B) is appended to self.
        """
        global sandwich_hits, sandwich_failures

        SNF = Sandwich(A,B)
        Gap = SNF.gap()

        # crucial that SNF is a LatticePolytope (or something else with a good hash),
        # not a Polyhedron (which has a poor hash)
        if SNF not in self[Gap]:
            self[Gap][SNF] = [A,B]
            sandwich_failures += 1
        else:
            sandwich_hits += 1

    def __repr__(self):
        return f'{self.__class__.__name__} with keys {sorted(self)}'


class SandwichFactory_with_diskcache_Index(SandwichFactory):
    r"""
    gap -> SandwichStorage
    """
    def __init__(self, dirname):
        try:
            import diskcache
        except ImportError:
            raise ImportError('Use !pip install diskcache')

        self._dirname = dirname
        self._sandwich_cache = diskcache.Cache(self._dirname + f'_invariants')

    def __missing__(self, key):
        mapping_factory = make_diskcache_Index_factory(self._dirname + f'_gap{key}')
        value = SandwichStorage_with_diskcache_Cache(mapping_factory, cache=self._sandwich_cache)
        self[key] = value
        return value

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dirname!r}) with keys {sorted(self)}'


def new_sandwich_factory(m, Delta, extremal, dirname=None):

    # Using https://github.com/mina86/pygtrie (https://pygtrie.readthedocs.io/en/latest/#pygtrie.Trie)
    # seemed promising, but unfortunately it always eagerly uses the whole key
    # when creating a new node (in _set_node).
    # (Our SandwichStorage does that only when we overwrite an item, which
    # we never do here.)
    #from pygtrie import Trie
    #sandwich_factory = defaultdict(Trie)

    if dirname is None:
        sandwich_factory = SandwichFactory()
    else:
        dirname += f'_m{m}_Delta{Delta}'
        if extremal:
            dirname += '_ext'
        sandwich_factory = SandwichFactory_with_diskcache_Index(dirname)

    for A,B in prepare_sandwiches(m,Delta):
        sandwich_factory.append_sandwich(A,B)
    return sandwich_factory


def sandwich_factory_statistics(sf):
    logging.info("Maximum gap in sandwiches: %d",max(sf.keys()))
    logging.info("Number of sandwiches: %d",sum([len(sf[Gap]) for Gap in sf.keys() if Gap!=0]))
    if 0 in sf.keys():
        logging.info("Number of polytopes found: %d", len(sf[0]))
    logging.info(f"Sandwich normal form hits: {sandwich_hits}, failures: {sandwich_failures}")
    logging.info(50*"-")


def delta_classification(m, Delta, extremal, dirname=None):
    """
        runs the sandwich factory algorithm and classifies all centrally symmetric m-dimensional lattice polytopes with largest determinant equal to Delta
        extremal is a Boolean parameter determining whether the whole classification is sought [extremal=false], or only the classification of the extremal examples attaining h(Delta,m) [extremal=true]
    """
    sf = new_sandwich_factory(m, Delta, extremal, dirname=dirname)
    maxGap = max(sf.keys())
    
    # set the known lower bound for h(Delta,m) by Lee et al. 
    if (extremal):
        cmax = m^2 - m + 1 *2*m*Delta
    
    while maxGap > 0:
        
        sandwich_factory_statistics(sf)
        
        for A, B in sf[maxGap].values():

            for v in B.vertices(): # pick a vertex of B which is not in A
                if v not in A[1]:
                    break
            
            blow_up_of_A = Polyhedron(list(A[1].vertices()) + [vector(v)] + [-vector(v)])	## this uses that all points in B are "Delta-ok" for A
            half_of_blow_up_of_A = break_symmetry(blow_up_of_A,m)
            reduction_of_B = Polyhedron([z for z in B.integral_points() if (vector(z) != vector(v) and vector(z) != -vector(v))])
            
            newA = [half_of_blow_up_of_A,blow_up_of_A]
            red_sand = reduce_sandwich(newA,B,Delta)
            if (extremal):
                if (red_sand.integral_points_count() >= cmax):
                    sf.append_sandwich(newA, red_sand)
                    npts_blow_up = blow_up_of_A.integral_points_count()
                    if (npts_blow_up > cmax):
                        cmax = npts_blow_up
                if (reduction_of_B.integral_points_count() >= cmax):
                    sf.append_sandwich(A, reduction_of_B)
            else:
                sf.append_sandwich(newA, red_sand)
                sf.append_sandwich(A, reduction_of_B)

        del sf[maxGap]
        maxGap = max(sf.keys())

    sandwich_factory_statistics(sf)
        
    result = []
    for A,B in sf[0].values():
        result.append(A[1])	## only store the polytope in A

    return result


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
    

