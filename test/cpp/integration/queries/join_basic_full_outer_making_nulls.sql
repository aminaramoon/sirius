select n.n_nationkey, r.r_regionkey
from nation n full outer join region r on n.n_nationkey = r.r_regionkey;
