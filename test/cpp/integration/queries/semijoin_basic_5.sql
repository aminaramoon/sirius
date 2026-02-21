select n.n_name from nation n semi join customer c
on n.n_nationkey = c.c_nationkey;
