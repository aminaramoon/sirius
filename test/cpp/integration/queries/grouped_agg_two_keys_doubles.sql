select c_nationkey, c_mktsegment, min(C_ACCTBAL), max(C_ACCTBAL), sum(C_ACCTBAL),
count(C_ACCTBAL) from customer group by c_nationkey, c_mktsegment;
