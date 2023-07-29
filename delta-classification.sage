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

from sage.geometry.lattice_polytope import LatticePolytope
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
        Invariant: first symmetric function
        """
        row_sums = Partition(sorted(sum(self._B_vertex_facet_pairing_matrix.columns()), reverse=True))
        column_sums = Partition(sorted(sum(self._B_vertex_facet_pairing_matrix.rows()), reverse=True))
        #print(row_sums, column_sums)
        return row_sums, column_sums

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
        row_sums = Partition(sorted(sum(self._A_vertex_B_facet_pairing_matrix.columns()), reverse=True))
        column_sums = Partition(sorted(sum(self._A_vertex_B_facet_pairing_matrix.rows()), reverse=True))
        return row_sums, column_sums

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

    @cached_method(do_pickle=True)
    def _key_func_LLP_permutation_normal_form(self):
        "FIXME: This is apparently NOT a normal form"
        #return self._LLP.normal_form(algorithm='palp')  # fastest of all, but crashes for dim > 3.
        #PNF = self._LLP_vertex_facet_pairing_matrix.permutation_normal_form(check=False)  # faster
        PNF = self._LLP._palp_PM_max(check=False)       # slower (before https://github.com/sagemath/sage/pull/35997), much faster (after)
        PNF.set_immutable()
        return PNF

    def _key_func_LLP_palp_native_normal_form(self):
        return self._LLP.normal_form(algorithm='palp_native')

    def key_funcs(self):
        return (self._key_func_dimensions,
                self._key_func_B_partitions,
                #self._key_func_A_vertex_B_facet_partitions,
                #self._key_func_B_permutation_normal_form,
                #self._key_func_A_vertex_B_facet_permutation_normal_form,
                #self._key_func_LLP_permutation_normal_form
                self._key_func_LLP_palp_native_normal_form)

    @staticmethod
    def key_costs():
        return (0, 1, 100)

    def __len__(self):
        return len(self.key_funcs())

    def __getitem__(self, i):
        r"""
        Return the components of the key for the trie
        """
        return self.key_funcs()[i]()

    def __eq_noninvariant__(self, other):
        return self._A == other._A and self._B == other._B

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
        - identify a unique candidate for ``key``, in which case ``(key_prefix, (key, value))`` is returned
        - show that ``key`` is not in ``self``, in which case ``(key_prefix, None)`` is returned
        """
        key_prefix = ()
        key_iter = iter(key)
        while True:
            mapping = self._key_prefix_to_mapping(key_prefix)
            try:
                item = mapping[key_prefix]
            except KeyError:
                return key_prefix, None
            else:
                if item != 'not_unique':
                    return key_prefix, item

            # possible improvement: insisting on a unique candidate is too much when the next key element
            # is too expensive. When the subtrie has <= THRESHOLD candidates, it may be faster to invert:
            # loop through all candidates and do the fast non-invariant check.
            if (cost := key.key_costs()[len(key_prefix)]) > 1:
                if len(next_mapping := self._key_prefix_to_mapping(key_prefix + (None,))) <= cost:
                    for same_prefix, item in next_mapping.items():
                        if item != 'not_unique':
                            if item[0].__eq_noninvariant__(key):
                                return key_prefix + (item[0][len(key_prefix)],), item

            try:
                key_prefix = key_prefix + (next(key_iter),)
            except StopIteration:
                assert False, 'path cannot end with not_unique'

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __getitem__(self, key):
        key_prefix, item = self._sufficient_key_prefix(key)
        if item is None:
            raise KeyError(key)
        candidate_key, candidate_value = item
        if len(key_prefix) == len(key):
            return candidate_value
        if candidate_key == key:
            return candidate_value
        raise KeyError(key)  # f'{key!r}; _sufficient_key_prefix returned {key_prefix=} {item=}')

    def __setitem__(self, key, value):
        key_prefix, item = self._sufficient_key_prefix(key)
        mapping = self._key_prefix_to_mapping(key_prefix)
        if item is not None:
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


def layered_polytope_from_sandwich(A,B):
    """ 3*B is embedded into height 0, two copies of 3*A are embedded into heights 1 and -1.
        Then, one generates a polytope based on these three layers at heights -1,0 and 1
        Note: If A and B are centrally symmetric, then the resulting polytope is centrally symmetric as well.
    """ 
    middleLayer = [tuple(3*vector(v))+(0,) for v in B.vertices()]
    upperLayer = [tuple(3*vector(v))+(1,) for v in A[1].vertices()]
    lowerLayer = [tuple(3*vector(v))+(-1,) for v in A[1].vertices()]
    return Polyhedron(middleLayer+upperLayer+lowerLayer)

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


## def sandwich_invariants(A, B):
##     return sum(sum(LatticePolytope(A[1].vertices_list()).vertex_facet_pairing_matrix())), sum(sum(LatticePolytope(B.vertices_list()).vertex_facet_pairing_matrix()))

class SandwichFactory_with_diskcache_Index(defaultdict):
    r"""
    gap -> SandwichStorage
    """
    def __init__(self, dirname):
        self._dirname = dirname

    def __missing__(self, key):
        value = SandwichStorage(mapping_factory=make_diskcache_Index_factory(self._dirname + f'_gap{key}'))
        self[key] = value
        return value

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dirname!r}) with keys {sorted(self)}'


def append_sandwich(sf, A, B):
    """
        If no affine unimodular image of the sandwich (A,B) is in the sandwich factory sf,
        the sandwich (A,B) is appended to sf.
    """
    global sandwich_hits, sandwich_failures

    Gap = B.integral_points_count() - A[1].integral_points_count()
    SNF = Sandwich(A,B)

    # crucial that SNF is a LatticePolytope (or something else with a good hash),
    # not a Polyhedron (which has a poor hash)
    if SNF not in sf[Gap]:
        sf[Gap][SNF] = [A,B]
        #print(SNF)
        #print(sandwich_invariants(A, B))
        sandwich_failures += 1
    else:
        sandwich_hits += 1


def new_sandwich_factory(m, Delta, extremal, dirname=None):

    # Using https://github.com/mina86/pygtrie (https://pygtrie.readthedocs.io/en/latest/#pygtrie.Trie)
    # seemed promising, but unfortunately it always eagerly uses the whole key
    # when creating a new node (in _set_node).
    # (Our SandwichStorage does that only when we overwrite an item, which
    # we never do here.)
    #from pygtrie import Trie
    #sandwich_factory = defaultdict(Trie)

    if dirname is None:
        sandwich_factory = defaultdict(SandwichStorage)
    else:
        dirname += f'_m{m}_Delta{Delta}'
        if extremal:
            dirname += '_ext'
        sandwich_factory = SandwichFactory_with_diskcache_Index(dirname)

    for A,B in prepare_sandwiches(m,Delta):
        append_sandwich(sandwich_factory,A,B)
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
                    append_sandwich(sf,newA,red_sand)
                    npts_blow_up = blow_up_of_A.integral_points_count()
                    if (npts_blow_up > cmax):
                        cmax = npts_blow_up
                if (reduction_of_B.integral_points_count() >= cmax):
                    append_sandwich(sf,A,reduction_of_B)
            else:
                append_sandwich(sf,newA,red_sand)
                append_sandwich(sf,A,reduction_of_B)
            
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
    

