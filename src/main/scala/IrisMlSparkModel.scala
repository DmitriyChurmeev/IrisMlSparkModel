import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Row}

object IrisMlSparkModel extends App {

  override val appName: String = "IrisMlSpark"

  def main(args: Array[String]) = {

    val irisDataFrame = getData("src/main/resources/IRIS.csv")

    val (trainingData, testData) = {
      val split = irisDataFrame.randomSplit(Array(0.8, 0.2))
      (split(0), split(1))
    }

    // Индексация типов данных для классификации (Iris-setosa\Iris-versicolor\Iris-virginica)
    val indexerTypeOfIris = new StringIndexer()
      .setInputCol("iris_type")
      .setOutputCol("label")

    // И прогнозирования типа по данным
    val randomIrisClassifier = new RandomForestClassifier()
      .setFeaturesCol("iris_data")

    val pipeline = new Pipeline().setStages(Array(indexerTypeOfIris, randomIrisClassifier))
    val trainValidation = buildTrainValidationSplit(pipeline, randomIrisClassifier)

    // Создание модели с обученными данными
    trainingData.cache()
    val model = trainValidation.fit(trainingData)

    // Получение результат на основе тестовых данных
    testData.cache()
    val resultsTestData = model.transform(testData)

    /**
     * Тестирование модели
     * предсказанной prediction и с фактической меткой в label
     */
    import spark.implicits._

    val predAndLabels = resultsTestData
      .select("prediction", "label")
      .map { case Row(prediction: Double, label: Double) => (prediction, label)
      }.rdd

    val multiclassMetrics = new MulticlassMetrics(predAndLabels)

    println(s"F1 Measure (эффективность) ${multiclassMetrics.weightedFMeasure}")
    println(s"Recall (полнота) ${multiclassMetrics.weightedRecall}")
    println(s"Precision (точность) ${multiclassMetrics.weightedPrecision}")

  }

  /**
   * Возвращает параметизированый классификатор для проверки точности модели
   */
  def buildParamGridBuilder(classifier: RandomForestClassifier): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(classifier.maxDepth, Array(1, 3, 9))
      .addGrid(classifier.numTrees, Array(10, 40, 60))
      .addGrid(classifier.impurity, Array("gini", "entropy"))
      .build()
  }

  def buildTrainValidationSplit(pipeline: Pipeline, classifier: RandomForestClassifier): TrainValidationSplit = {

    val paramGrid = buildParamGridBuilder(classifier)
    new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
  }


  /**
   * Получение DataFrame (Vectors данных и тип) из файла
   * @param path путь до файла
   * @return DataFrame
   */
  def getData(path: String): DataFrame = {
    val irisDf = spark.sparkContext
      .textFile(path)
      .mapPartitionsWithIndex {
        (idx, iter) => if (idx == 0) iter.drop(1) else iter
      }

    val irisRdd = irisDf
      .flatMap {
        text =>
          text.split("\n").toList.map(_.split(",")).collect {
            case Array(sepal_length, sepal_width, petal_length, petal_width, species) =>
              (Vectors.dense(sepal_length.toDouble, sepal_width.toDouble, petal_length.toDouble, petal_width.toDouble), species)
          }
      }

    spark.createDataFrame(irisRdd).toDF("iris_data", "iris_type")
  }
}
