select n.n_regionkey from nation n
semi join customer c on n.n_nationkey = c.c_custkey;
