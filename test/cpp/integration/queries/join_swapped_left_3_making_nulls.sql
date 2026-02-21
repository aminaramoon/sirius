select n.n_name, c.c_custkey, c.c_name
from customer c left join nation n on n.n_nationkey = c.c_custkey;
