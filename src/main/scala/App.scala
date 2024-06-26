import org.apache.spark.sql.SparkSession

trait App {
  val appName: String

  lazy val spark = createSession(appName)

  private def createSession(appName: String) = {
    SparkSession.builder()
      .appName(appName)
      .master("local[*]")
      .getOrCreate()
  }
}
