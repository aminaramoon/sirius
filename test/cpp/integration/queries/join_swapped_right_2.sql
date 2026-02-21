select n.n_regionkey, c.c_nationkey, c.c_custkey, c.c_name
from customer c right join nation n on n.n_nationkey = c.c_nationkey;
