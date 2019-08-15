<h1>Ascend's Autonomous Dataflows - Lab Session 1 </h1>

<h2>Before you begin:</h2>


1. Participants will work in teams of 2, but each participant will have their own login.
2. Each team will be assigned a Data Service where the team's own dataflow(s) will be built.
3. Please follow the instructions below and copy-and-paste code snippets to build out the components of your dataflow.

<h3>Step 1: Logging into the Ascend environment</h3>

*   Click here to log into the Ascend environment:  [http://phlai-1.ascend.io/](http://phlai-1.ascend.io/)
*   Use your team login and password provided on the separate sheet.
*   Click ‘**Skip Tutorial**’ once you login as this will take you to the default interactive tutorial.


<h3>Step 2: Navigate to your Data Service</h3>

*   Once you login you will immediately be dropped into your own data service  Your data service will have the name such as “Team1”.
*   Click “_+ NEW DATAFLOW_”


<h3>Step 3: Create a new dataflow</h3>

*   Enter the name: **Insurance Claims Analysis**
*   Click _CREATE_


<h3>Step 4: Connect to a Data Feed</h3>

*   In the upper left-hand portion of the GUI: Click ADD -> DATA FEED -> CONNECT
*   Select the “**Insurance Training Datafeed”**
*   Click _SUBSCRIBE_
*   Enter the name: **“Training dataset”**
*   Click _CREATE_


<h3>Step 5: Create the Feature Prep & Clean transform</h3>

*   In the upper left-hand portion of the GUI: Click ADD -> TRANSFORM
*   Enter the name: **“Feature Prep & Clean”**
*   Under “_QUERY - HOW WILL YOU SPECIFY THE QUERY_” select **”SQL”**
*   Enter the following SQL code in the SQL Editor:

        ```
        select
          i.KIDSDRIV as KIDSDRIV,
          i.AGE as AGE,
          i.HOMEKIDS as HOMEKIDS,
          i.MSTATUS as MSTATUS,
          i.CAR_USE as CAR_USE,
          i.CAR_TYPE as CAR_TYPE,
          i.CAR_AGE AS AGE_FEATURE,
          REPLACE(substr(i.INCOME, 2), ',', '') as INCOME_FEATURE,
          i.CLAIM_FLAG as label
        FROM
          Training_dataset AS i
        ```
        
*   Click _CREATE_


<h3>Step 6: Create the Randomize transform</h3>

*   In the upper left-hand portion of the GUI: Click ADD -> TRANSFORM
*   Enter the name: **“Randomize”**
*   Under “_QUERY - HOW WILL YOU SPECIFY THE QUERY_” select **“PySpark”**
*   Under “_WHICH UPSTREAM DATASET WILL BE INPUT[0]?_” select **“Feature_Prep__Clean”**
*   Under “_HOW WILL PARTITIONS BE HANDLED?_” select **“Reduce all Partitions to One”**
*   Replace the existing PySpark code with the following PySpark code in the PySpark Function Editor:

        ```
        from typing import List
        from pyspark.sql import DataFrame, SparkSession
        from pyspark.sql.functions import rand,row_number

        def transform(spark_session: SparkSession, inputs: List[DataFrame]) -> DataFrame:
        	allData = inputs[0]
        	return allData.orderBy(rand())
        ```

*   Click CREATE


<h3>Step 7: Create the Model Training transform</h3>

*   In the upper left-hand portion of the GUI: Click ADD -> TRANSFORM
*   Enter the name: **“Model Training”**
*   Under “_QUERY - HOW WILL YOU SPECIFY THE QUERY_” select **“PySpark”**
*   Under “_WHICH UPSTREAM DATASET WILL BE INPUT[0]?_” select **“Randomize”**
*   Under “_HOW WILL PARTITIONS BE HANDLED?_” select **“Map all Partitions One-to-One”**
*   Replace the existing PySpark code with the following PySpark code in the PySpark Function Editor:

        ```
        from typing import List
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
        from pyspark.ml.feature import VectorAssembler,StringIndexer
        from pyspark.sql import DataFrame, SparkSession
        from pyspark.sql.functions import col
        from pyspark.sql.types import IntegerType

        import time

        def transform(spark_session: SparkSession, inputs: List[DataFrame]) -> DataFrame:
        	allData = inputs[0]

        	# Define feature columns
        	featureColumns = allData.schema.names
        	featureColumns.remove("label")
        

        	# Convert numerical column types to integers
        	allData = allData.withColumn("KIDSDRIV", allData["KIDSDRIV"].cast(IntegerType()))
        	allData = allData.withColumn("INCOME_FEATURE", allData["INCOME_FEATURE"].cast(IntegerType()))
        	allData = allData.withColumn("AGE", allData["AGE"].cast(IntegerType()))
        	allData = allData.withColumn("AGE_FEATURE", allData["AGE_FEATURE"].cast(IntegerType()))
        	allData = allData.withColumn("label", allData["label"].cast(IntegerType()))

        	allData = allData.where(col("KIDSDRIV").isNotNull())
        	allData = allData.where(col("INCOME_FEATURE").isNotNull())
        	allData = allData.where(col("AGE").isNotNull())
        	allData = allData.where(col("AGE_FEATURE").isNotNull())
        	allData = allData.where(col("MSTATUS").isNotNull())
        	allData = allData.where(col("CAR_USE").isNotNull())
        	allData = allData.where(col("CAR_TYPE").isNotNull())
        	allData = allData.where(col("label").isNotNull())

        	# Index categorical features and create a feature vector
        	categoricalFeatures = ["MSTATUS", "CAR_USE", "CAR_TYPE"]
        	indexers = [StringIndexer(inputCol = name, outputCol = name+"_FEATURE") for name in categoricalFeatures]
        	featureColumns += [i.getOutputCol() for i in indexers]
        	featureColumns = [x for x in featureColumns if x not in categoricalFeatures]
        	va = VectorAssembler(inputCols=featureColumns, outputCol="features")

        	# Define ML model to train
        	lr = LogisticRegression(maxIter=1_000, regParam=0.01, elasticNetParam=0.8)

        	# Link all PySpark components up into an ML Pipeline
        	pipeline = Pipeline(stages=indexers + [va, lr])

        	# Split data to training and validation sets
        	training, test = allData.randomSplit([0.7, 0.3])
        	# Since we are streaming, we need to establish a base case for PySpark start up
        	if training.count() == 0:
        		training = spark_session.createDataFrame([[0 for i in range(len(allData.schema.names))]], training.schema)

        	# Train
        	model = pipeline.fit(training)

        	# Validate/Test
        	predictions = model.transform(test)

        	return predictions.select(featureColumns + ["label", "prediction"])
        ```

*   Click _CREATE_


<h3>Step 8: Create the Model Accuracy transform</h3>

*   In the upper left-hand portion of the GUI: Click ADD -> TRANSFORM
*   Enter the name: **“Model Accuracy”**
*   Under “_QUERY - HOW WILL YOU SPECIFY THE QUERY_” select **“PySpark”**
*   Under “_WHICH UPSTREAM DATASET WILL BE INPUT[0]?_” select **“Model_Training”**
*   Under “_HOW WILL PARTITIONS BE HANDLED?_” select **“Map all Partitions One-to-One”**
*   Enter the following PySpark code in the PySpark Function Editor:

        ```
        from typing import List
        from pyspark.sql import DataFrame, SparkSession
        from pyspark.sql.types import FloatType

        def transform(spark_session: SparkSession, inputs: List[DataFrame]) -> DataFrame:
        	Model_Training = inputs[0]
        	Model_Training.createTempView("predictions")
        	total = spark_session.sql("""SELECT COUNT(*) AS total_count FROM predictions""").collect()
        	total_val = total[0].total_count

        	correct = spark_session.sql("""SELECT COUNT(*) AS correct_count FROM predictions WHERE label=prediction""").collect()
        	correct_val = correct[0].correct_count

        	if total_val != 0:
        		accuracy = 100.0 * correct_val / total_val
        	else:
        		accuracy = 0.0

        	return spark_session.createDataFrame([accuracy], FloatType())
        ```
        
*   Click _CREATE_


<h3>Step 9: Iterate on the Feature Prep & Clean transform</h3>

*   Click on the **“Feature Prep & Clean”** transform
*   Click _EDIT_
*   In the SQL Editor add the following column (near line 8): 

    ```
    i.CLM_FREQ as CLM_FREQ,
    ```

*   Click _UPDATE_
*   Click on the **“Model Training”** transform
*   Click _EDIT_
*   In the **Model Training** component add the following between line 20 and line 21 using the PySpark Editor:

    ```
    allData = allData.withColumn("CLM_FREQ", allData["CLM_FREQ"].cast(IntegerType()))
    ```
*   Also, in the PySPark Editor add the following at line 27:

```
	allData = allData.where(col("CLM_FREQ").isNotNull())

```
*   Click _UPDATE_

<h3>Step 10: Swap out Classification Models</h3>

*   Click on the **"Model Training"** transform
*   Click _EDIT_
*   Using the PySpark editor, navigate to Line 44, where we previously defined our **"LogisticRegression"** model. 
*   Here, we will try training a **"RandomForestClassifier"** ([PySpark documentation](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#random-forest-classifier)). 
*   Replace the code on Line 44 with the following:
```
lr = RandomForestClassifier(numTrees=10)
```
*   You will also need to make sure that you have imported the **"RandomForestClassifier"** model, so on Line 3, add **",RandomForestClassifier"** to the end. 
*   Line 3 should now look like this:
```
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel,RandomForestClassifier
```
*   Click _UPDATE_
*   You have now trained an ML classifier with a Random Forest model!

<h3>Extra Credit: Tuning and Experimenting with other models</h3>

*   **Tuning**: Tuning our models is really simple now. We can adjust specific parameters in each PySpark model such as changing **"numTrees=10"** in Step 10 to **"numTrees=20"**.

*   **Experimentation**: If you want to go above and beyond this lab and experiment with other classifiers, find your favorite [Pyspark classifier](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html) and follow the steps in Step 10 to import your new classifier, initializing it with the right parameters, and clicking _UPDATE_!
