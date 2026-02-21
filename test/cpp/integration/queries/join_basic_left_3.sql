select n.n_name, c.c_custkey, c.c_name
from nation n left join customer c on n.n_nationkey = c.c_nationkey;
