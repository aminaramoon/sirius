select n.n_nationkey from nation n semi join region r on n.n_regionkey = r.r_regionkey;
