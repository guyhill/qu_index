import qu_index as qu_index
import pandas as pd

data = qu_index.load_microdata_csv("testdata\\products.csv")
indexes = qu_index.calculate(data)
print(indexes)

ref_indexes = pd.read_csv("testdata\\indexes.csv", sep = ",")
ref_indexes.rename(columns = {"Index": "reference_qu_index"}, inplace = True)
indexes = pd.merge(indexes, ref_indexes, on = ("Verslagperiode", "Productgroep"))
indexes["absdiff"] = (indexes["qu_index"] - indexes["reference_qu_index"]).abs()
indexes["reldiff"] = indexes["absdiff"] / indexes["reference_qu_index"]

print(indexes)
print(f"Maximum absolute difference between calculated QU index and reference: {indexes['absdiff'].max():.2e}")
print(f"Maximum relative difference between calculated QU index and reference: {indexes['reldiff'].max():.2e}")
