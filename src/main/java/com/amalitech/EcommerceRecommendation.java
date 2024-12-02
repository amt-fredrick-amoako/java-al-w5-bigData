package com.amalitech;

import org.apache.spark.sql.*;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.*;

public class EcommerceRecommendation {
    public static void main(String[] args) {
        // 1. Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("Recommendation Engine")
                .master("local[*]") // Run locally with all available cores
                .getOrCreate();

        // 2. Define Schema for Input Data
        StructType schema = new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("product_id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("rating", DataTypes.IntegerType, false, Metadata.empty())
        });

        // 3. Load Dataset
        Dataset<Row> purchases = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv("src/main/resources/ml-latest-small/purchases.csv"); // Path to the dataset file

        // 4. Display Dataset
        System.out.println("Input Data:");
        purchases.show();

        // 5. Train ALS Model
        ALS als = new ALS()
                .setMaxIter(10)                 // Number of iterations
                .setRegParam(0.1)               // Regularization parameter
                .setUserCol("user_id")          // Column for users
                .setItemCol("product_id")       // Column for products
                .setRatingCol("rating");        // Column for ratings

        ALSModel model = als.fit(purchases);

        // 6. Generate Recommendations for Users
        Dataset<Row> userRecommendations = model.recommendForAllUsers(3); // 3 recommendations per user
        System.out.println("User Recommendations:");
        userRecommendations.show(false);

        // 7. Generate Recommendations for Products
        Dataset<Row> productRecommendations = model.recommendForAllItems(3); // 3 recommendations per product
        System.out.println("Product Recommendations:");
        productRecommendations.show(false);

        // 8. Stop Spark Session
        spark.stop();
    }
}

