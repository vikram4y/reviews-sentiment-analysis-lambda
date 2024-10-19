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

    private static final String REVIEWS_TABLE = "ProductReviews";
    private static final String ANALYSIS_TABLE = "ProductReviewAnalysis";

    private DynamoDB dynamoDB;
    private AmazonComprehend comprehendClient;

    @Override
    public Map<String, Object> handleRequest(Map<String, String> input, Context context) {
        initializeDynamoDB();
        initializeComprehend();

        String productId = input.get("product_id");
        if (isInvalidProductId(productId)) {
            return Map.of("result", "Error: product_id is missing.");
        }

        List<String> reviews = fetchProductReviews(productId);
        if (reviews.isEmpty()) {
            return Map.of("result", "Product not found!");
        }

        Map<String, Object> analysisResult = analyzeReviews(reviews);

        storePrecomputedResult(productId, analysisResult);

        return analysisResult;
    }

    private void initializeDynamoDB() {
        AmazonDynamoDB client = AmazonDynamoDBClientBuilder.standard().build();
        dynamoDB = new DynamoDB(client);
    }

    private void initializeComprehend() {
        comprehendClient = AmazonComprehendClientBuilder.standard().build();
    }

    private boolean isInvalidProductId(String productId) {
        return productId == null || productId.isEmpty();
    }

    private List<String> fetchProductReviews(String productId) {
        Table table = dynamoDB.getTable(REVIEWS_TABLE);
        QuerySpec querySpec = new QuerySpec().withKeyConditionExpression("product_id = :v_id")
                .withValueMap(Map.of(":v_id", productId));

        ItemCollection<QueryOutcome> items = table.query(querySpec);
        List<String> reviews = new ArrayList<>();
        items.forEach(item -> reviews.add(item.getString("review_text")));
        return reviews;
    }

    private Map<String, Object> analyzeReviews(List<String> reviews) {
        Map<String, Integer> sentimentCounts = initializeSentimentCounts();
        Map<String, Double> sentimentConfidences = initializeSentimentConfidences();
        Map<String, Map<String, Integer>> aspectSentiments = new HashMap<>();

        int shortReviewsCount = 0;
        int longReviewsCount = 0;
        List<String> keyPhrases = new ArrayList<>();

        for (String review : reviews) {
            DetectSentimentResult sentimentResult = detectSentiment(review);
            String sentiment = sentimentResult.getSentiment().toUpperCase();
            incrementSentimentCount(sentimentCounts, sentiment);
            updateSentimentConfidence(sentimentConfidences, sentiment, sentimentResult.getSentimentScore());

            analyzeKeyPhrases(review, sentiment, aspectSentiments, keyPhrases);

            if (review.length() < 50) {
                shortReviewsCount++;
            } else if (review.length() > 200) {
                longReviewsCount++;
            }
        }

        return buildFinalResult(reviews.size(), sentimentCounts, sentimentConfidences, aspectSentiments, shortReviewsCount, longReviewsCount, keyPhrases);
    }

    private DetectSentimentResult detectSentiment(String review) {
        DetectSentimentRequest sentimentRequest = new DetectSentimentRequest()
                .withText(review)
                .withLanguageCode("en");

        return comprehendClient.detectSentiment(sentimentRequest);
    }

    private void analyzeKeyPhrases(String review, String sentiment, Map<String, Map<String, Integer>> aspectSentiments, List<String> keyPhrases) {
        DetectKeyPhrasesRequest keyPhrasesRequest = new DetectKeyPhrasesRequest()
                .withText(review)
                .withLanguageCode("en");

        DetectKeyPhrasesResult keyPhrasesResult = comprehendClient.detectKeyPhrases(keyPhrasesRequest);

        for (KeyPhrase phrase : keyPhrasesResult.getKeyPhrases()) {
            String keyPhrase = phrase.getText().toLowerCase();
            keyPhrases.add(keyPhrase);

            aspectSentiments.computeIfAbsent(keyPhrase, k -> initializeSentimentCounts());
            incrementSentimentCount(aspectSentiments.get(keyPhrase), sentiment);
        }
    }

    private Map<String, Integer> initializeSentimentCounts() {
        return new HashMap<>(Map.of("POSITIVE", 0, "NEGATIVE", 0, "NEUTRAL", 0, "MIXED", 0));
    }

    private Map<String, Double> initializeSentimentConfidences() {
        return new HashMap<>(Map.of("POSITIVE", 0.0, "NEGATIVE", 0.0, "NEUTRAL", 0.0, "MIXED", 0.0));
    }

    private void incrementSentimentCount(Map<String, Integer> sentimentCounts, String sentiment) {
        sentimentCounts.put(sentiment, sentimentCounts.get(sentiment) + 1);
    }

    private void updateSentimentConfidence(Map<String, Double> sentimentConfidences, String sentiment, SentimentScore sentimentScore) {
        double maxConfidence = getMaxSentimentConfidence(sentimentScore);
        if (!Double.isNaN(maxConfidence)) {
            sentimentConfidences.put(sentiment, sentimentConfidences.get(sentiment) + maxConfidence);
        }
    }

    private double getMaxSentimentConfidence(SentimentScore sentimentScore) {
        return Math.max(Math.max(sentimentScore.getPositive(), sentimentScore.getNegative()),
                Math.max(sentimentScore.getNeutral(), sentimentScore.getMixed()));
    }

    private Map<String, Object> buildFinalResult(int totalReviews, Map<String, Integer> sentimentCounts, Map<String, Double> sentimentConfidences,
                                                 Map<String, Map<String, Integer>> aspectSentiments, int shortReviewsCount, int longReviewsCount, List<String> keyPhrases) {
        Map<String, Object> result = new HashMap<>();
        result.put("total_reviews", totalReviews);
        result.put("sentiment_percentages", calculateSentimentPercentages(sentimentCounts, totalReviews));
        result.put("average_sentiment_confidence", calculateAverageConfidence(sentimentCounts, sentimentConfidences));
        result.put("short_reviews_count", shortReviewsCount);
        result.put("long_reviews_count", longReviewsCount);
        result.put("top_key_phrases", getTopKeyPhrases(keyPhrases));
        result.put("top_aspect_based_sentiments", getTopAspectSentiments(aspectSentiments));

        return result;
    }

    private Map<String, Double> calculateSentimentPercentages(Map<String, Integer> sentimentCounts, int totalReviews) {
        return sentimentCounts.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> (e.getValue() / (double) totalReviews) * 100));
    }

    private Map<String, Double> calculateAverageConfidence(Map<String, Integer> sentimentCounts, Map<String, Double> sentimentConfidences) {
        return sentimentCounts.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> {
                    int count = e.getValue();
                    return count > 0 ? sentimentConfidences.get(e.getKey()) / count : 0.0;
                }));
    }

    private List<String> getTopKeyPhrases(List<String> keyPhrases) {
        return keyPhrases.stream()
                .limit(5)  // Change ranking logic as needed
                .collect(Collectors.toList());
    }

    private Map<String, Map<String, Integer>> getTopAspectSentiments(Map<String, Map<String, Integer>> aspectSentiments) {
        return aspectSentiments.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue().get("POSITIVE"), a.getValue().get("POSITIVE")))  // Sort by POSITIVE sentiment
                .limit(5)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    private void storePrecomputedResult(String productId, Map<String, Object> analysisResult) {
        Table table = dynamoDB.getTable(ANALYSIS_TABLE);
        table.putItem(new PutItemSpec().withItem(new Item()
                .withPrimaryKey("product_id", productId)
                .withMap("review_analysis", analysisResult)));
    }
}
