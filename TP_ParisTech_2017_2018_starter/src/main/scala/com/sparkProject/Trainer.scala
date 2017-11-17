package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

   val df: DataFrame = spark
     .read
     .option("header", true)  // Use first line of all files as header
     .option("inferSchema", "true") // Try to infer the data types of each column
      .load("../prepared_trainingset")

    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    df.show()


    /** TF-IDF **/

      // 1er stage
    val tokenizer = new RegexTokenizer().setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // 2eme stage
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")


    // 3eme stage
    val countVec = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("cvModel")

    // 4eme stage
    val idf = new IDF()
    .setInputCol("cvModel")
      .setOutputCol("tfidf")


    /** VECTOR ASSEMBLER **/

    // 5eme stage
    val countryIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    // 6eme stage
    val currencyIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** MODEL **/


    // 7eme stage

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    // 8eme stage
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7,0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVec, idf, countryIndexer, currencyIndexer, assembler, lr))

    /** TRAINING AND GRID-SEARCH **/

    // Create a df called "training" and another one called "test" from the df that i separate into
    // training and test in the following order 90%,10%

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))

    val param_grid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVec.minDF,  Array[Double](55, 75, 95))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(param_grid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training)

    val df_with_predictions = model.transform(test)

    println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_with_predictions))

    df_with_predictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite().save("TP4_5_Spark_Final_Model")


  }
}
