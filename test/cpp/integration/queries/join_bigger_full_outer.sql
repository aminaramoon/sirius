select l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_partkey,
o.o_orderkey, o.o_totalprice, o.o_custkey, o_comment
from lineitem l full outer join orders o on l.l_orderkey = o.o_orderkey
order by l.l_orderkey, l.l_linenumber;
