select c_nationkey, min(C_NAME), max(C_NAME), count(C_NAME) from customer
group by c_nationkey;
