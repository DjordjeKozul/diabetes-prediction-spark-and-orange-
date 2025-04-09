from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder,VectorAssembler

import findspark
findspark.init()

spark= SparkSession.builder.appName("Klasifikacija").getOrCreate();
data=spark.read.csv("C:/Users/Djole/Desktop/diabetes_dataset.csv",header=True,inferSchema=True,nullValue="",nanValue="")
data=data.dropna(how="any")


Glucose=data.select("Glucose").agg({"Glucose":"min"})
Glucose.show()
glukoza_count=data.select("Glucose").count()
glukoza=data.select("Glucose").take(glukoza_count)

binovi=np.arange(0,210,5)
glukoza_vrednosti=[x[0] for x in glukoza]
plt.hist(glukoza_vrednosti,binovi)
plt.show()

kolone_za_enkodiranje=data.columns[-5:-1]

enkoder=OneHotEncoder(inputCols=kolone_za_enkodiranje,outputCols=[col+"_enkodirano" for col in kolone_za_enkodiranje],dropLast=True)
data_enkodirano=enkoder.fit(data).transform(data)
data_enkodirano.show(n=3,truncate=False)

kolone=data_enkodirano.columns[:-9]+data_enkodirano.columns[-5:]
data=data_enkodirano.select(kolone)
kolone_za_skaliranje=data.columns[:-5]

asembler=VectorAssembler(inputCols=kolone_za_skaliranje,outputCol="features_")
data_skalirano=asembler.transform(data)
data_skalirano=data_skalirano.select(data_skalirano.columns[-6:])

data_train,data_test=data_skalirano.randomSplit([.8,.2])

scaler=StandardScaler(withMean=True,withStd=True,inputCol="features_",outputCol="Skalirane kolone")
scaler_model=scaler.fit(data_train)

data_train_skalirano=scaler_model.transform(data_train)
data_test_skalirano=scaler_model.transform(data_test)

asembler_krajnji=VectorAssembler(inputCols=data_test_skalirano.columns[1:-2]+[data_test_skalirano.columns[-1]],outputCol="features")
data_train=asembler_krajnji.transform(data_train_skalirano)
data_test=asembler_krajnji.transform(data_test_skalirano)

data_train=data_train.select(["features","Outcome"])
data_test=data_test.select(["features","Outcome"])

#--------------    Model i treniranje  ------------------------------------

lr=LogisticRegression(featuresCol="features",labelCol="Outcome")
model=lr.fit(data_train)
predikcije=model.transform(data_test)

evaluator_f1= MulticlassClassificationEvaluator(labelCol="Outcome",metricName="f1")
evaluator_acc= MulticlassClassificationEvaluator(labelCol="Outcome",metricName="accuracy")
evaluator_AUC= MulticlassClassificationEvaluator(labelCol="Outcome")

print(evaluator_acc.evaluate(predikcije))
print(evaluator_f1.evaluate(predikcije))
print(evaluator_AUC.evaluate(predikcije))

# Test primer
jedan_primer = data_test.limit(1)
predikcija_ = model.transform(jedan_primer)
predikcija_.select("features", "Outcome", "prediction", "probability").show(truncate=False)
spark.stop()