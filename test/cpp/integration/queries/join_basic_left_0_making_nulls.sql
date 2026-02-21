select n.n_nationkey, n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name
from nation n left join customer c on n.n_nationkey = c.c_custkey;
