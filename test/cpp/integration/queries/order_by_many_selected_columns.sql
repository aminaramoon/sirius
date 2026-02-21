select l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity
from lineitem order by l_suppkey;
