package com.amalitech;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Hello world!
 *
 */
public class RecommendationEngine
{
    public static void main( String[] args )
    {
        SparkSession sparkSession = SparkSession.builder()
                .appName("Recommendation Engine with ALS")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> ratings = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/ml-latest-small/ratings.csv");

//        ratings.show();
       Dataset<Row>[] splits = ratings.randomSplit(new double[] { 0.8, 0.2 }, 42);
       Dataset<Row> trainingData = splits[0];
       Dataset<Row> testData = splits[1];

        ALS als = new ALS()
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating")
                .setColdStartStrategy("drop");

        ParamGridBuilder paramGrid = new ParamGridBuilder()
                .addGrid(als.rank(), new int[]{5, 10, 15})
                .addGrid(als.maxIter(), new int[]{5, 10, 20})
                .addGrid(als.regParam(), new double[]{0.01, 0.1, 0.5});

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(als)
                .setEvaluator(new RegressionEvaluator()
                        .setMetricName("rmse")
                        .setLabelCol("rating")
                        .setPredictionCol("prediction"))
                .setEstimatorParamMaps(paramGrid.build())
                .setNumFolds(5);

        ALSModel model = (ALSModel) crossValidator.fit(trainingData).bestModel();

        Dataset<Row> predictions = model.transform(trainingData);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Square Error (RMSE) on test data; " + rmse);

        Dataset<Row> userRecommendations = model.recommendForAllUsers(10);
        Dataset<Row> itemRecommendations = model.recommendForAllItems(10);

        System.out.println("User Recommendations:");
        userRecommendations.show(false);

        System.out.println("Item Recommendations:");
        itemRecommendations.show(false);

        sparkSession.stop();
    }
}
