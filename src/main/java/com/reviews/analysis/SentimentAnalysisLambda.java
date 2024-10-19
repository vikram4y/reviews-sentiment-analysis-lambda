package com.reviews.analysis;

import com.amazonaws.services.comprehend.AmazonComprehend;
import com.amazonaws.services.comprehend.AmazonComprehendClientBuilder;
import com.amazonaws.services.comprehend.model.*;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.document.DynamoDB;
import com.amazonaws.services.dynamodbv2.document.Table;
import com.amazonaws.services.dynamodbv2.document.ItemCollection;
import com.amazonaws.services.dynamodbv2.document.QueryOutcome;
import com.amazonaws.services.dynamodbv2.document.spec.QuerySpec;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;

import java.util.*;

public class SentimentAnalysisLambda implements RequestHandler<Map<String, String>, Map<String, Object>> {

    private static final String TABLE_NAME = "ProductReviews";
    private DynamoDB dynamoDB;

    @Override
    public Map<String, Object> handleRequest(Map<String, String> input, Context context) {
        initializeDynamoDB();
        String productId = input.get("product_id");
        Table table = dynamoDB.getTable(TABLE_NAME);

        // Fetch reviews from DynamoDB
        QuerySpec querySpec = new QuerySpec().withKeyConditionExpression("product_id = :v_id")
                .withValueMap(new HashMap<String, Object>() {{
                    put(":v_id", productId);
                }});

        ItemCollection<QueryOutcome> items = table.query(querySpec);
        List<String> reviews = new ArrayList<>();

        items.forEach(item -> reviews.add(item.getString("review_text")));

        // Amazon Comprehend client
        AmazonComprehend comprehendClient = AmazonComprehendClientBuilder.standard().build();

        // Data structures for result aggregation
        Map<String, Integer> sentimentCount = new HashMap<>();
        sentimentCount.put("POSITIVE", 0);
        sentimentCount.put("NEGATIVE", 0);
        sentimentCount.put("NEUTRAL", 0);
        sentimentCount.put("MIXED", 0);

        Map<String, Double> sentimentConfidence = new HashMap<>();
        sentimentConfidence.put("POSITIVE", 0.0);
        sentimentConfidence.put("NEGATIVE", 0.0);
        sentimentConfidence.put("NEUTRAL", 0.0);
        sentimentConfidence.put("MIXED", 0.0);

        Map<String, Map<String, Integer>> aspectSentiment = new HashMap<>();
        List<String> keyPhrases = new ArrayList<>();
        int totalReviews = reviews.size();
        int shortReviewsCount = 0;
        int longReviewsCount = 0;

        // Analyze each review
        for (String review : reviews) {
            DetectSentimentRequest sentimentRequest = new DetectSentimentRequest()
                    .withText(review)
                    .withLanguageCode("en");

            DetectSentimentResult sentimentResult = comprehendClient.detectSentiment(sentimentRequest);
            String sentiment = sentimentResult.getSentiment().toUpperCase();
            SentimentScore sentimentScore = sentimentResult.getSentimentScore();

            sentimentCount.put(sentiment, sentimentCount.get(sentiment) + 1);
            sentimentConfidence.put(sentiment, sentimentConfidence.get(sentiment) + getMaxSentimentConfidence(sentimentScore));

            DetectKeyPhrasesRequest keyPhrasesRequest = new DetectKeyPhrasesRequest()
                    .withText(review)
                    .withLanguageCode("en");

            DetectKeyPhrasesResult keyPhrasesResult = comprehendClient.detectKeyPhrases(keyPhrasesRequest);
            for (KeyPhrase phrase : keyPhrasesResult.getKeyPhrases()) {
                String keyPhrase = phrase.getText().toLowerCase();
                keyPhrases.add(keyPhrase);

                // Update aspect sentiment for the dynamic key phrases (aspects)
                if (!aspectSentiment.containsKey(keyPhrase)) {
                    aspectSentiment.put(keyPhrase, new HashMap<String, Integer>() {{
                        put("POSITIVE", 0);
                        put("NEGATIVE", 0);
                        put("NEUTRAL", 0);
                        put("MIXED", 0);
                    }});
                }

                Map<String, Integer> sentimentMap = aspectSentiment.get(keyPhrase);
                sentimentMap.put(sentiment, sentimentMap.get(sentiment) + 1);
            }

            // Analyze review length
            if (review.length() < 50) {
                shortReviewsCount++;
            } else if (review.length() > 200) {
                longReviewsCount++;
            }
        }

        // Calculate sentiment percentages
        Map<String, Double> sentimentPercentages = new HashMap<>();
        for (Map.Entry<String, Integer> entry : sentimentCount.entrySet()) {
            sentimentPercentages.put(entry.getKey(), (entry.getValue() / (double) totalReviews) * 100);
        }

        // Calculate average confidence for each sentiment
        for (Map.Entry<String, Double> entry : sentimentConfidence.entrySet()) {
            sentimentConfidence.put(entry.getKey(), entry.getValue() / sentimentCount.get(entry.getKey()));
        }

        // Prepare the final output
        Map<String, Object> output = new HashMap<>();
        output.put("total_reviews", totalReviews);
        output.put("sentiment_percentages", sentimentPercentages);
        output.put("average_sentiment_confidence", sentimentConfidence);
        output.put("key_phrases", keyPhrases);
        output.put("aspect_based_sentiment", aspectSentiment);
        output.put("short_reviews_count", shortReviewsCount);
        output.put("long_reviews_count", longReviewsCount);

        return output;
    }

    private double getMaxSentimentConfidence(SentimentScore sentimentScore) {
        return Math.max(Math.max(sentimentScore.getPositive(), sentimentScore.getNegative()),
                Math.max(sentimentScore.getNeutral(), sentimentScore.getMixed()));
    }

    private void initializeDynamoDB() {
        AmazonDynamoDB client = AmazonDynamoDBClientBuilder.standard().build();
        this.dynamoDB = new DynamoDB(client);
    }
}

