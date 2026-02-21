select c.c_custkey, c.c_name from customer c
semi join nation n on n.n_nationkey = c.c_nationkey;
