select avg(n_regionkey), avg(n_nationkey), n_name,
max(cast(n_nationkey as Decimal(18,2)))
from nation group by n_regionkey, n_name;