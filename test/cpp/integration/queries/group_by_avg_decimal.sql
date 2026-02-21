select l_returnflag, avg(l_quantity), avg(l_discount)
from lineitem group by l_returnflag;
