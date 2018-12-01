# Konfigurasi Spark
import os
import sys

# 1. Mengeset variabel yang menyimpan lokasi di mana Spark diinstal
spark_path = "E:\\5115100002pil\spark-2.3.0-bin-hadoop2.7\\spark-2.3.0-bin-hadoop2.7"

# 2. Menentukan environment variable SPARK_HOME
os.environ['SPARK_HOME'] = spark_path

# 3. Simpan lokasi winutils.exe sebagai environment variable HADOOP_HOME
os.environ['HADOOP_HOME'] = "E:\\5115100002pil\\hadoop-2.8.3"

# 4. Lokasi Python yang dijalankan --> punya Anaconda
#    Apabila Python yang diinstall hanya Anaconda, maka tidak perlu menjalankan baris ini.
os.environ['PYSPARK_PYTHON'] = sys.executable

# 5. Konfigurasi path library PySpark
sys.path.append(spark_path + "/bin")
sys.path.append(spark_path + "/python")
sys.path.append(spark_path + "/python/pyspark/")
sys.path.append(spark_path + "/python/lib")
sys.path.append(spark_path + "/python/lib/pyspark.zip")
sys.path.append(spark_path + "/python/lib/py4j-0.10.6-src.zip")

# 6. Import library Spark
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import PCA
import math
from pyspark.sql import Row


# Setting konfigurasi (opsional)
conf = SparkConf()
conf.set("spark.executor.memory", "6g")
conf.set("spark.cores.max", "4")

sc = SparkContext("local", conf=conf)
#    Apabila berhasil, maka ketika sc di-print akan mengeluarkan nilai <pyspark.context.SparkContext object>
print(sc)

spark = SparkSession.builder.appName('collegeTree').getOrCreate()

# 1. Data preparation
sql = SQLContext(sc)
df = sql.read \
        .format("com.databricks.spark.csv") \
        .option("inferSchema", "true") \
        .load("datasets/SUSY.csv")
    
#df.schema
#df.show()

vecCols = df.columns
    for delCol in ['_c0']:
        vecCols.remove(delCol)
#print vecCols
assembler = VectorAssembler(inputCols=vecCols, outputCol='features')
data_assem = assembler.transform(df).select('_c0','features')
#data_assem.show(10)
data_feed= data_assem.select('_c0','features')

newCols = df.columns
	for newDel in ['_c0','_c1','_c2','_c3','_c4','_c5','_c6','_c7','_c8']:
		newCols.remove(newDel)
assembler2 = VectorAssembler(inputCols=newCols, outputCol='features')
dsemm= assembler.transform(df).select('_c0','features')

# #PCA
# data_feed2=df.columns
#     for dCol in ['_c0']:
#         data_feed2.remove(dCol)
# assembler2 = VectorAssembler(inputCols=data_feed2, outputCol='features')
# print assembler2
# dsemm= assembler.transform(df).select('features')
# dsemm.show()

# pca = PCA(k=3, inputCol=dsemm, outputCol="pcaFeatures")
# model = pca.fit(df)
# result = model.transform(df).select('pcaFeatures')
# result.show(truncate=False)

train_data, test_data = data_feed.randomSplit([0.7,0.3])

dtc = DecisionTreeClassifier(labelCol='_c0', featuresCol='features')
rfc = RandomForestClassifier(labelCol='_c0', featuresCol='features')
gbt = GBTClassifier(labelCol='_c0', featuresCol='features')

dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)
#rfc_preds.printSchema()

binary_eval = BinaryClassificationEvaluator(labelCol='_c0')

print('DTC prediction AUC:', binary_eval.evaluate(dtc_preds))
print('RFC prediction AUC:', binary_eval.evaluate(rfc_preds))
print('GBT prediction AUC:', binary_eval.evaluate(gbt_preds))

acc_eval = MulticlassClassificationEvaluator(labelCol='_c0', metricName='accuracy')
print('DTC prediction accuracy:', acc_eval.evaluate(dtc_preds))
print('RFC prediction accuracy:', acc_eval.evaluate(rfc_preds))
print('GBT prediction accuracy:', acc_eval.evaluate(gbt_preds))


lr = LogisticRegression(labelCol='_c0', featuresCol='features', maxIter=10)
lrModel = lr.fit(train_data)
predictions = lrModel.transform(test_data)
#predictions.printSchema()
evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
print('Logistic Regression AUC: ',binary_eval.evaluate(predictions))
print('Logistic Regression prediction accuracy:', acc_eval.evaluate(predictions))


#print('under area ROC: ',evaluator.evaluate(predictions))
#evaluator.getMetricName() print(lr.explainParams())
# Train a NaiveBayes model
#nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
## Chain labelIndexer, vecAssembler and NBmodel in a 
#pp = Pipeline(stages=[data_assem, assembler, nb])
#model = pp.fit(train_data)
#predictions = model.transform(test_data)
#evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
#                                              metricName="precision")
#accuracy = evaluator.evaluate(predictions)
#print "Model Accuracy: ", accuracy

model = NaiveBayes.train(train_data, 1.0)
predictionAndLabel= test_data.map(lambda p: (model.predict(p.features),p.label))
accuracy= 1.0*predictionAndLabel.filter(lambda(x,v):x == v).count()/test_data.count()

##rfc combine gbt
#rfc_2 = RandomForestClassifier(labelCol='_c0', featuresCol='features', numTrees=100)
#gbt_2=gbt_preds
#rfc_preds_2 = gbt_2.fit(train_data).transform(test_data)
#print('RFC_2 prediction AUC:', binary_eval.evaluate(rfc_preds_2))

#from pyspark.mllib.recommendation import MatrixFactorizationModel
#model_path = os.path.join('BIGData', 'models', 'fp2')
### Save and load model
#model.save(sc, model_path)
#same_model = MatrixFactorizationModel.load(sc, model_path)