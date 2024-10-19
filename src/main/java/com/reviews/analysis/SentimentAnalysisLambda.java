package com.reviews.analysis;

import com.amazonaws.services.comprehend.AmazonComprehend;
import com.amazonaws.services.comprehend.AmazonComprehendClientBuilder;
import com.amazonaws.services.comprehend.model.*;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.document.*;
import com.amazonaws.services.dynamodbv2.document.spec.PutItemSpec;
import com.amazonaws.services.dynamodbv2.document.spec.QuerySpec;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;

import java.util.*;
import java.util.stream.Collectors;

public class SentimentAnalysisLambda implements RequestHandler<Map<String, String>, Map<String, Object>> {

    private static final String REVIEW_TABLE = "ProductReviews";
    private static final String ANALYSIS_TABLE = "ProductReviewAnalysis";
    private static final int SHORT_REVIEW_THRESHOLD = 50;
    private static final int LONG_REVIEW_THRESHOLD = 200;
    private DynamoDB dynamoDB;

    @Override
    public Map<String, Object> handleRequest(Map<String, String> input, Context context) {
        initializeDynamoDB();

        String productId = input.get("product_id");
        if (productId == null || productId.isEmpty()) {
            return Map.of("result", "Error: product_id is missing.");
        }

        List<String> reviews = fetchReviews(productId);
        if (reviews.isEmpty()) {
            return Map.of("result", "Product not found!");
        }

        return analyzeReviews(productId, reviews);
    }

    private List<String> fetchReviews(String productId) {
        Table reviewTable = dynamoDB.getTable(REVIEW_TABLE);
        QuerySpec querySpec = new QuerySpec().withKeyConditionExpression("product_id = :v_id")
                .withValueMap(Map.of(":v_id", productId));

        ItemCollection<QueryOutcome> items = reviewTable.query(querySpec);
        List<String> reviews = new ArrayList<>();
        items.forEach(item -> reviews.add(item.getString("review_text")));

        return reviews;
    }

    private Map<String, Object> analyzeReviews(String productId, List<String> reviews) {
        AmazonComprehend comprehendClient = AmazonComprehendClientBuilder.standard().build();

        // Initialize counters and result structures
        Map<String, Integer> sentimentCount = initializeSentimentCount();
        Map<String, Double> sentimentConfidence = initializeSentimentConfidence();
        Map<String, Map<String, Integer>> aspectSentimentMap = new HashMap<>();
        List<String> keyPhrases = new ArrayList<>();
        int shortReviewCount = 0, longReviewCount = 0;

        // Process each review
        for (String review : reviews) {
            processSentiment(comprehendClient, review, sentimentCount, sentimentConfidence);
            processKeyPhrases(comprehendClient, review, keyPhrases, aspectSentimentMap);

            if (review.length() < SHORT_REVIEW_THRESHOLD) {
                shortReviewCount++;
            } else if (review.length() > LONG_REVIEW_THRESHOLD) {
                longReviewCount++;
            }
        }

        // Calculate final sentiment percentages and confidence averages
        int totalReviews = reviews.size();
        Map<String, Double> sentimentPercentages = calculateSentimentPercentages(sentimentCount, totalReviews);
        Map<String, Double> averageSentimentConfidence = calculateAverageConfidence(sentimentCount, sentimentConfidence);

        // Prepare output
        Map<String, Object> analysisResult = prepareOutput(
                totalReviews, sentimentPercentages, averageSentimentConfidence,
                shortReviewCount, longReviewCount, keyPhrases, aspectSentimentMap
        );

        // Store the precomputed analysis result in DynamoDB
        storeAnalysisResult(productId, analysisResult);

        return analysisResult;
    }

    private Map<String, Integer> initializeSentimentCount() {
        return new HashMap<>(Map.of("POSITIVE", 0, "NEGATIVE", 0, "NEUTRAL", 0, "MIXED", 0));
    }

    private Map<String, Double> initializeSentimentConfidence() {
        return new HashMap<>(Map.of("POSITIVE", 0.0, "NEGATIVE", 0.0, "NEUTRAL", 0.0, "MIXED", 0.0));
    }

    private void processSentiment(AmazonComprehend comprehendClient, String review,
                                  Map<String, Integer> sentimentCount, Map<String, Double> sentimentConfidence) {
        DetectSentimentRequest sentimentRequest = new DetectSentimentRequest()
                .withText(review)
                .withLanguageCode("en");

        DetectSentimentResult sentimentResult = comprehendClient.detectSentiment(sentimentRequest);
        String sentiment = sentimentResult.getSentiment().toUpperCase();
        sentimentCount.put(sentiment, sentimentCount.get(sentiment) + 1);

        double confidence = getMaxSentimentConfidence(sentimentResult.getSentimentScore());
        sentimentConfidence.put(sentiment, sentimentConfidence.get(sentiment) + confidence);
    }

    private void processKeyPhrases(AmazonComprehend comprehendClient, String review, List<String> keyPhrases,
                                   Map<String, Map<String, Integer>> aspectSentimentMap) {
        DetectKeyPhrasesRequest keyPhrasesRequest = new DetectKeyPhrasesRequest()
                .withText(review)
                .withLanguageCode("en");

        DetectKeyPhrasesResult keyPhrasesResult = comprehendClient.detectKeyPhrases(keyPhrasesRequest);
        for (KeyPhrase phrase : keyPhrasesResult.getKeyPhrases()) {
            String keyPhrase = phrase.getText().toLowerCase();
            keyPhrases.add(keyPhrase);
            aspectSentimentMap.putIfAbsent(keyPhrase, initializeSentimentCount());
        }
    }

    private double getMaxSentimentConfidence(SentimentScore sentimentScore) {
        return Collections.max(List.of(
                sentimentScore.getPositive(), sentimentScore.getNegative(),
                sentimentScore.getNeutral(), sentimentScore.getMixed()
        ));
    }

    private Map<String, Double> calculateSentimentPercentages(Map<String, Integer> sentimentCount, int totalReviews) {
        return sentimentCount.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> (entry.getValue() / (double) totalReviews) * 100
                ));
    }

    private Map<String, Double> calculateAverageConfidence(Map<String, Integer> sentimentCount, Map<String, Double> sentimentConfidence) {
        return sentimentConfidence.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> sentimentCount.get(entry.getKey()) > 0
                                ? entry.getValue() / sentimentCount.get(entry.getKey())
                                : 0.0
                ));
    }

    private Map<String, Object> prepareOutput(int totalReviews, Map<String, Double> sentimentPercentages,
                                              Map<String, Double> averageSentimentConfidence, int shortReviewCount,
                                              int longReviewCount, List<String> keyPhrases,
                                              Map<String, Map<String, Integer>> aspectSentimentMap) {
        Map<String, Object> output = new HashMap<>();
        output.put("total_reviews", totalReviews);
        output.put("sentiment_percentages", sentimentPercentages);
        output.put("average_sentiment_confidence", averageSentimentConfidence);
        output.put("short_reviews_count", shortReviewCount);
        output.put("long_reviews_count", longReviewCount);

        List<String> topKeyPhrases = keyPhrases.stream().limit(5).collect(Collectors.toList());
        output.put("top_key_phrases", topKeyPhrases);

        Map<String, Map<String, Integer>> topAspectSentiments = aspectSentimentMap.entrySet().stream()
                .sorted(Comparator.comparingInt(entry -> -entry.getValue().getOrDefault("POSITIVE", 0)))
                .limit(5)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        output.put("top_aspect_based_sentiments", topAspectSentiments);
        return output;
    }

    private void initializeDynamoDB() {
        AmazonDynamoDB client = AmazonDynamoDBClientBuilder.standard().build();
        this.dynamoDB = new DynamoDB(client);
    }

    private void storeAnalysisResult(String productId, Map<String, Object> analysisResult) {
        Table analysisTable = dynamoDB.getTable(ANALYSIS_TABLE);
        analysisTable.putItem(new PutItemSpec().withItem(new Item()
                .withPrimaryKey("product_id", productId)
                .withMap("review_analysis", analysisResult)));
    }
}
