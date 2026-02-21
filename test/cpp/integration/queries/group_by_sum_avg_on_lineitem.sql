select l_returnflag, l_linestatus, sum(l_quantity), avg(l_extendedprice), count(*)
from lineitem group by l_returnflag, l_linestatus;
