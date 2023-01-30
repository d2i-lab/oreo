from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import col, row_number,lit
from pyspark.sql.window import Window
from functools import reduce
from pyspark.sql import DataFrame


def add_bid(row, bids):
    row_dict = row.asDict()
    bid = bids[int(row_dict["idx"])]
    row_dict["bid"] = int(bid)
    return row_dict


class PartitionManager:
    def __init__(self, session, k):
        self.session = session
        self.k = k

    def load_parquet(self, root):
        dfs = []
        for i in range(self.k):
            path = "%s/bid=%d" % (root, i)
            df = self.session.read.parquet(path)
            dfs.append(df)
        return reduce(DataFrame.union, dfs)

    def load_parquet_bids(self, root, bids):
        dfs = []
        for bid in bids:
            path = "%s/bid=%d" % (root, bid)
            df = self.session.read.parquet(path)
            dfs.append(df)
        return reduce(DataFrame.union, dfs)

    def load_csv(self, file):
        return self.session.read.format("csv").option("header", "True").load(file)

    def reorg_with_labels(self, df, bids, out_dir):
        # Add bid column to dataframe
        rdd = df.rdd.map(lambda row : add_bid(row, bids))
        old_schema = df.schema
        new_schema = old_schema.add("bid", IntegerType())
        df_augmented = self.session.createDataFrame(rdd, new_schema)
        # Write to disk and partition by bid
        df_augmented.write.option("header", True) \
            .partitionBy("bid") \
            .mode("overwrite") \
            .parquet(out_dir)

    def load_and_reorg_with_labels(self, bids, in_dir, out_dir, csv=False):
        # Load data from csv
        if csv:
            df = self.load_csv(in_dir)
        else:
            df = self.load_parquet(in_dir)
        self.reorg_with_labels(df, bids, out_dir)

    def run_query(self, dir, bids, query):
        if len(bids) == 0:
            return
        df = self.load_parquet_bids(dir, bids)
        df.createOrReplaceTempView("tbl")
        ans = self.session.sql(query)
        ans.count()

