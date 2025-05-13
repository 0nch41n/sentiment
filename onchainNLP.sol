// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

/**
 * @title SentimentMiniLLM
 * @dev Simplified on-chain sentiment analyzer with context-awareness
 */
contract SentimentMiniLLM is AccessControl, Pausable {
    using Strings for uint256;

    // Roles
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant TRAINER_ROLE = keccak256("TRAINER_ROLE");
    bytes32 public constant FEEDBACK_PROVIDER = keccak256("FEEDBACK_PROVIDER");
    bytes32 public constant DATA_CONSUMER_ROLE = keccak256("DATA_CONSUMER_ROLE");

    // Constants
    uint256 public constant MAX_VOCAB_SIZE = 1024;
    uint256 public constant EMBEDDING_DIM = 24;
    uint256 public constant CONTEXT_DIM = 8;
    int256 public constant SCALE_FACTOR = 1000;
    uint256 public constant MAX_INPUT_TOKENS = 16;
    uint8 public constant CLASS_COUNT = 7;
    uint8 public constant MAX_FEEDBACK_PER_USER = 20;
    uint256 public constant FEEDBACK_COOLDOWN = 12 hours;
    uint8 public constant MAX_PHRASE_LENGTH = 4;
    uint256 public constant MAX_DOMAIN_COUNT = 10;

    // Sentiment classes
    uint8 public constant CLASS_VERY_NEGATIVE = 0;
    uint8 public constant CLASS_NEGATIVE = 1;
    uint8 public constant CLASS_SLIGHTLY_NEGATIVE = 2;
    uint8 public constant CLASS_NEUTRAL = 3;
    uint8 public constant CLASS_SLIGHTLY_POSITIVE = 4;
    uint8 public constant CLASS_POSITIVE = 5;
    uint8 public constant CLASS_VERY_POSITIVE = 6;

    // Sentiment flags
    uint8 public constant FLAG_POSITIVE = 1;
    uint8 public constant FLAG_NEGATIVE = 2;
    uint8 public constant FLAG_EMOTIONAL = 4;
    uint8 public constant FLAG_DOMAIN_SPECIFIC = 8;
    uint8 public constant FLAG_INTENSE = 16;
    uint8 public constant FLAG_AMBIGUOUS = 32;
    uint8 public constant FLAG_SARCASTIC = 64;
    uint8 public constant FLAG_CONTEXT_DEPENDENT = 128;

    // Semantic categories
    uint8 public constant CATEGORY_EMOTION = 0;
    uint8 public constant CATEGORY_ACTION = 1;
    uint8 public constant CATEGORY_CONCEPT = 2;
    uint8 public constant CATEGORY_OBJECT = 3;
    uint8 public constant CATEGORY_ATTRIBUTE = 4;
    uint8 public constant CATEGORY_RELATION = 5;
    uint8 public constant CATEGORY_QUANTITY = 6;
    uint8 public constant CATEGORY_TEMPORAL = 7;
    uint8 public constant CATEGORY_LOCATION = 8;

    // Domain types
    uint8 public constant DOMAIN_GENERAL = 0;
    uint8 public constant DOMAIN_FINANCE = 1;
    uint8 public constant DOMAIN_TECH = 2;
    uint8 public constant DOMAIN_HEALTH = 3;
    uint8 public constant DOMAIN_POLITICS = 4;
    uint8 public constant DOMAIN_ENTERTAINMENT = 5;
    uint8 public constant DOMAIN_EDUCATION = 6;
    uint8 public constant DOMAIN_SOCIAL = 7;
    uint8 public constant DOMAIN_TRAVEL = 8;
    uint8 public constant DOMAIN_FOOD = 9;

    // Enhanced token metadata
    struct TokenMetadata {
        int8 sentiment;
        uint8 flags;
        uint8 category;
        uint8 weight;
        uint8 domainRelevance;
        uint8 secondaryCategory;
        uint16 cooccurrenceCount;
        uint8 contextInfluence;
    }

    // Enhanced user context
    struct UserContext {
        uint32 lastInteraction;
        uint16 lastInputToken;
        uint8 feedbackCount;
        uint32 feedbackCooldown;
        uint16[3] recentTopicTokens;
        uint8[CLASS_COUNT] classHistory;
        uint8 primaryDomain;
        uint16 totalInteractions;
        int8 sentimentBias;
    }

    // Enhanced feedback with confidence and explanation
    struct Feedback {
        uint256[] inputTokens;
        uint8 predictedClass;
        uint8 actualClass;
        int128 confidence;
        uint32 timestamp;
        string explanation;
    }

    // Phrase data for multi-word recognition
    struct Phrase {
        string[] words;
        uint256 tokenId;
        uint8 primaryWordIndex;
    }

    // Domain-specific sentiment modifier
    struct DomainModifier {
        int8[CLASS_COUNT] classBias;
        uint8 intensity;
    }

    // Storage
    mapping(uint256 => string) internal tokenVocab;
    mapping(string => uint256) internal wordToToken;
    mapping(uint256 => TokenMetadata) internal tokenMetadata;
    mapping(uint256 => int256[EMBEDDING_DIM]) internal tokenEmbeddings;
    mapping(uint256 => int256[CONTEXT_DIM]) internal contextEmbeddings;
    mapping(uint8 => int256[EMBEDDING_DIM]) internal classWeights;
    mapping(uint8 => int256[CONTEXT_DIM]) internal classContextWeights;
    mapping(address => UserContext) internal userContexts;
    mapping(address => Feedback[]) internal userFeedback;
    mapping(uint256 => uint256) internal tokenUsageCount;
    mapping(uint256 => mapping(uint256 => uint16)) internal tokenCooccurrence;
    mapping(uint8 => DomainModifier) internal domainModifiers;
    mapping(string => Phrase) internal phrases;
    mapping(uint256 => uint8) internal tokenDomainStrength;

    // Statistical tracking
    uint256 internal vocabSize;
    uint256 internal totalClassifications;
    uint256 internal correctPredictions;
    mapping(uint8 => uint256) internal classDistribution;
    uint256 internal phraseCount;

    // Events
    event SentimentClassified(
        address indexed user, 
        uint8 classId, 
        int256 confidence, 
        string input, 
        uint8 domain
    );
    event VocabUpdated(uint256 count, address indexed trainer);
    event WeightsUpdated(uint8 classCount, address indexed trainer);
    event FeedbackSubmitted(
        address indexed user, 
        uint256[] inputTokens, 
        uint8 actualClass, 
        uint8 predictedClass
    );
    event VocabProposed(
        address indexed proposer, 
        string word, 
        int8 sentiment, 
        uint8 flags, 
        uint8 category
    );
    event PhraseAdded(string indexed phrase, uint256 tokenId);
    event ModelAccuracyUpdated(uint256 totalPredictions, uint256 correctPredictions);
    event DomainModifierUpdated(uint8 domainId, uint8 intensity);

    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(ADMIN_ROLE, msg.sender);
        _setupRole(TRAINER_ROLE, msg.sender);
        _setupRole(FEEDBACK_PROVIDER, msg.sender);
        _setupRole(DATA_CONSUMER_ROLE, msg.sender);

        // Initialize domain modifiers with neutral values
        for (uint8 i = 0; i < MAX_DOMAIN_COUNT; i++) {
            domainModifiers[i].intensity = 5; // Moderate intensity
        }
    }

    // ========== INTERNAL HELPERS ========== //

    /**
     * @dev Convert string to lowercase
     */
    function _toLower(string memory str) internal pure returns (string memory) {
        bytes memory bStr = bytes(str);
        bytes memory bLower = new bytes(bStr.length);
        for (uint256 i = 0; i < bStr.length; i++) {
            if (uint8(bStr[i]) >= 65 && uint8(bStr[i]) <= 90) {
                bLower[i] = bytes1(uint8(bStr[i]) + 32);
            } else {
                bLower[i] = bStr[i];
            }
        }
        return string(bLower);
    }

    /**
     * @dev Find partial match for a word
     */
    function _findPartialMatch(string memory word) internal view returns (uint256) {
        bytes memory wordBytes = bytes(word);
        if (wordBytes.length < 3) return 0;

        uint256 bestToken = 0;
        uint256 bestScore = 0;
        uint256 bestUsageCount = 0;

        for (uint256 i = 0; i < vocabSize; i++) {
            bytes memory vocabWord = bytes(tokenVocab[i]);
            if (vocabWord.length >= wordBytes.length) {
                // Check for prefix match (higher priority)
                bool isPrefixMatch = true;
                uint256 prefixScore = 0;

                for (uint256 j = 0; j < wordBytes.length && j < vocabWord.length; j++) {
                    if (vocabWord[j] == wordBytes[j]) {
                        prefixScore++;
                    } else {
                        isPrefixMatch = false;
                        break;
                    }
                }

                if (isPrefixMatch && prefixScore > bestScore) {
                    bestScore = prefixScore;
                    bestToken = i;
                    bestUsageCount = tokenUsageCount[i];
                }

                // If not a prefix match, check for partial match
                if (!isPrefixMatch) {
                    uint256 partialScore = 0;
                    for (uint256 j = 0; j < wordBytes.length; j++) {
                        for (uint256 k = 0; k < vocabWord.length; k++) {
                            if (wordBytes[j] == vocabWord[k]) {
                                partialScore++;
                                break;
                            }
                        }
                    }

                    // Require higher threshold for non-prefix matches
                    if (partialScore > wordBytes.length * 7 / 10 && // 70% match
                        (partialScore > bestScore || 
                         (partialScore == bestScore && tokenUsageCount[i] > bestUsageCount))) {
                        bestScore = partialScore;
                        bestToken = i;
                        bestUsageCount = tokenUsageCount[i];
                    }
                }
            }
        }

        // Ensure match is good enough
        return bestScore >= wordBytes.length / 2 ? bestToken : 0;
    }

    /**
     * @dev Check if a sequence of words forms a known phrase
     */
    function _checkForPhrase(string[] memory words, uint256 startIndex) internal view returns (uint256, uint8) {
        if (startIndex >= words.length) return (0, 0);

        // Try phrases of different lengths
        for (uint8 length = MAX_PHRASE_LENGTH; length > 1; length--) {
            if (startIndex + length > words.length) continue;

            // Build the phrase
            string memory phraseStr = words[startIndex];
            for (uint8 i = 1; i < length; i++) {
                phraseStr = string(abi.encodePacked(phraseStr, " ", words[startIndex + i]));
            }

            // Look up the phrase
            uint256 tokenId = wordToToken[phraseStr];
            if (tokenId > 0) {
                return (tokenId, length);
            }
        }

        return (0, 0);
    }

    /**
     * @dev Calculate token similarity with improved logic and context awareness
     */
    function _calculateTokenSimilarity(
        uint256 tokenA, 
        uint256 tokenB, 
        bool includeContext
    ) internal view returns (int256) {
        if (tokenA >= vocabSize || tokenB >= vocabSize) return 0;

        int256 similarity = 0;

        // Semantic embedding similarity
        for (uint256 i = 0; i < EMBEDDING_DIM; i++) {
            similarity += tokenEmbeddings[tokenA][i] * tokenEmbeddings[tokenB][i] / SCALE_FACTOR;
        }

        // Add context embedding similarity if requested
        if (includeContext) {
            int256 contextSimilarity = 0;
            for (uint256 i = 0; i < CONTEXT_DIM; i++) {
                contextSimilarity += contextEmbeddings[tokenA][i] * contextEmbeddings[tokenB][i] / SCALE_FACTOR;
            }
            similarity += contextSimilarity / 2; // Weight context less than semantic meaning
        }

        // Category bonus
        TokenMetadata memory metaA = tokenMetadata[tokenA];
        TokenMetadata memory metaB = tokenMetadata[tokenB];

        if (metaA.category == metaB.category) {
            similarity += 150; // Primary category match bonus
        } else if (metaA.secondaryCategory == metaB.category || 
                  metaA.category == metaB.secondaryCategory) {
            similarity += 75; // Secondary category match bonus
        }

        // Domain relevance bonus
        if (metaA.domainRelevance == metaB.domainRelevance && metaA.domainRelevance > 0) {
            similarity += 100; // Same domain bonus
        }

        // Cooccurrence bonus
        if (tokenCooccurrence[tokenA][tokenB] > 0) {
            similarity += int256(uint256(tokenCooccurrence[tokenA][tokenB])) * 5; // Words that appear together often
        }

        // Sentiment polarity adjustment
        if ((metaA.sentiment > 0 && metaB.sentiment > 0) || 
            (metaA.sentiment < 0 && metaB.sentiment < 0)) {
            similarity += 50; // Same sentiment direction bonus
        } else if (metaA.sentiment != 0 && metaB.sentiment != 0) {
            similarity -= 30; // Opposing sentiment penalty
        }

        return similarity;
    }

    /**
     * @dev Determine primary domain for a set of input tokens
     */
    function _determinePrimaryDomain(uint256[] memory tokens) internal view returns (uint8) {
        uint256[MAX_DOMAIN_COUNT] memory domainScores;

        // Calculate domain scores
        for (uint256 i = 0; i < tokens.length; i++) {
            uint8 domain = tokenMetadata[tokens[i]].domainRelevance;
            uint8 strength = tokenDomainStrength[tokens[i]];

            if (domain < MAX_DOMAIN_COUNT && strength > 0) {
                domainScores[domain] += uint256(strength);
            }
        }

        // Find domain with highest score
        uint8 primaryDomain = DOMAIN_GENERAL;
        uint256 highestScore = domainScores[DOMAIN_GENERAL];

        for (uint8 i = 1; i < MAX_DOMAIN_COUNT; i++) {
            if (domainScores[i] > highestScore) {
                highestScore = domainScores[i];
                primaryDomain = i;
            }
        }

        return primaryDomain;
    }

    /**
     * @dev Update token cooccurrence statistics
     */
    function _updateCooccurrences(uint256[] memory tokens) internal {
        for (uint256 i = 0; i < tokens.length; i++) {
            for (uint256 j = i + 1; j < tokens.length; j++) {
                // Avoid overflow
                if (tokenCooccurrence[tokens[i]][tokens[j]] < type(uint16).max) {
                    tokenCooccurrence[tokens[i]][tokens[j]]++;
                }
                if (tokenCooccurrence[tokens[j]][tokens[i]] < type(uint16).max) {
                    tokenCooccurrence[tokens[j]][tokens[i]]++;
                }
            }
        }
    }

    /**
     * @dev Apply domain-specific modifiers to classification scores
     */
    function _applyDomainModifiers(
        int256[CLASS_COUNT] memory scores, 
        uint8 domain
    ) internal view returns (int256[CLASS_COUNT] memory) {
        if (domain == DOMAIN_GENERAL) return scores;

        DomainModifier memory domainMod = domainModifiers[domain];
        if (domainMod.intensity == 0) return scores;

        for (uint8 c = 0; c < CLASS_COUNT; c++) {
            int256 modifierEffect = int256(int8(domainMod.classBias[c])) * int256(uint256(domainMod.intensity)) * 10;
            scores[c] += modifierEffect;
        }

        return scores;
    }

    /**
     * @dev Apply user context to scoring
     */
    function _applyUserContext(
        int256[CLASS_COUNT] memory scores,
        UserContext memory context
    ) internal pure returns (int256[CLASS_COUNT] memory) {
        // Apply sentiment bias
        if (context.sentimentBias != 0) {
            for (uint8 c = 0; c < CLASS_COUNT; c++) {
                // Positive bias boosts positive classes, reduces negative ones
                if (c > CLASS_NEUTRAL) { // Positive classes
                    scores[c] += int256(context.sentimentBias) * 20;
                } else if (c < CLASS_NEUTRAL) { // Negative classes
                    scores[c] -= int256(context.sentimentBias) * 20;
                }
            }
        }

        // Apply class history - slight boost to previously seen classes
        if (context.totalInteractions > 0) {
            for (uint8 c = 0; c < CLASS_COUNT; c++) {
                uint8 classHistory = context.classHistory[c];
                if (classHistory > 0) {
                    scores[c] += int256(uint256(classHistory)) * 5;
                }
            }
        }

        return scores;
    }

    // ========== CORE CLASSIFICATION ========== //

    /**
     * @dev Classify sentiment with enhanced algorithm
     */
    function classifySentiment(uint256[] calldata inputTokens)
        external
        whenNotPaused
        returns (uint8 classId, int256 confidence, uint8 domain)
    {
        require(inputTokens.length > 0 && inputTokens.length <= MAX_INPUT_TOKENS, "Invalid input length");
        require(vocabSize > 0, "Vocabulary not initialized");

        // Create input string for event
        string memory inputString;
        for (uint256 i = 0; i < inputTokens.length; i++) {
            require(inputTokens[i] < vocabSize, "Invalid token");
            tokenUsageCount[inputTokens[i]]++;
            inputString = string(abi.encodePacked(
                inputString, 
                i == 0 ? "" : " ", 
                tokenVocab[inputTokens[i]]
            ));
        }

        // Update cooccurrence statistics
        _updateCooccurrences(inputTokens);

        // Determine primary domain for this input
        domain = _determinePrimaryDomain(inputTokens);

        // Calculate weighted embeddings
        int256[EMBEDDING_DIM] memory aggregatedEmbedding;
        int256[CONTEXT_DIM] memory aggregatedContext;
        int256 totalSentiment;
        uint256 totalWeight;

        // Process each token
        for (uint256 i = 0; i < inputTokens.length; i++) {
            TokenMetadata memory meta = tokenMetadata[inputTokens[i]];
            uint256 weight = uint256(meta.weight);

            // Context influence affects weighting
            if (meta.contextInfluence > 0) {
                weight = weight * uint256(meta.contextInfluence);
            }

            totalWeight += weight;

            // Add to embeddings
            for (uint256 j = 0; j < EMBEDDING_DIM; j++) {
                aggregatedEmbedding[j] += tokenEmbeddings[inputTokens[i]][j] * int256(weight);
            }

            // Add to context embeddings
            for (uint256 j = 0; j < CONTEXT_DIM; j++) {
                aggregatedContext[j] += contextEmbeddings[inputTokens[i]][j] * int256(weight);
            }

            // Add to sentiment with scaling by weight
            totalSentiment += int256(meta.sentiment) * int256(weight);
        }

        // Normalize embeddings
        if (totalWeight > 0) {
            for (uint256 j = 0; j < EMBEDDING_DIM; j++) {
                aggregatedEmbedding[j] = aggregatedEmbedding[j] / int256(totalWeight);
            }

            for (uint256 j = 0; j < CONTEXT_DIM; j++) {
                aggregatedContext[j] = aggregatedContext[j] / int256(totalWeight);
            }

            totalSentiment = totalSentiment / int256(totalWeight);
        }

        // Calculate class scores
        int256[CLASS_COUNT] memory scores;
        UserContext storage context = userContexts[msg.sender];

        for (uint8 c = 0; c < CLASS_COUNT; c++) {
            // Semantic embedding match
            for (uint256 j = 0; j < EMBEDDING_DIM; j++) {
                scores[c] += aggregatedEmbedding[j] * classWeights[c][j] / SCALE_FACTOR;
            }

            // Context embedding match
            for (uint256 j = 0; j < CONTEXT_DIM; j++) {
                scores[c] += aggregatedContext[j] * classContextWeights[c][j] / SCALE_FACTOR;
            }

            // Add raw sentiment score with increased weight
            scores[c] += totalSentiment * 15;

            // Add context from previous interaction
            if (context.lastInputToken > 0 && context.lastInteraction > block.timestamp - 1 hours) {
                int256 similarity = _calculateTokenSimilarity(
                    inputTokens[0], 
                    context.lastInputToken,
                    true // Include context embeddings
                );
                scores[c] += similarity / 10;
            }

            // Consider topic continuity
            for (uint8 t = 0; t < 3; t++) {
                if (context.recentTopicTokens[t] > 0) {
                    for (uint256 i = 0; i < inputTokens.length; i++) {
                        int256 topicSimilarity = _calculateTokenSimilarity(
                            inputTokens[i],
                            context.recentTopicTokens[t],
                            false // Skip context embeddings here
                        );
                        scores[c] += topicSimilarity / 20;
                    }
                }
            }
        }

        // Apply domain-specific modifiers
        scores = _applyDomainModifiers(scores, domain);

        // Apply user context
        scores = _applyUserContext(scores, context);

        // Find maximum score and calculate softmax-like probabilities
        int256 maxScore = scores[0];
        uint8 maxClass = 0;

        for (uint8 c = 1; c < CLASS_COUNT; c++) {
            if (scores[c] > maxScore) {
                maxScore = scores[c];
                maxClass = c;
            }
        }

        // Calculate confidence using normalized scores
        int256 totalScore = 0;
        for (uint8 c = 0; c < CLASS_COUNT; c++) {
            // Normalize scores relative to max score
            scores[c] = scores[c] - maxScore + 1000; // Shift to positive range
            totalScore += scores[c];
        }

        // Calculate confidence (0-1000)
        confidence = totalScore > 0 ? (scores[maxClass] * SCALE_FACTOR / totalScore) : int256(0);

        // Update user context
        context.lastInteraction = uint32(block.timestamp);
        context.lastInputToken = uint16(inputTokens[0]);

        // Update primary domain if strong domain detected
        if (domain != DOMAIN_GENERAL) {
            context.primaryDomain = domain;
        }

        // Circular buffer for recent topics
        context.recentTopicTokens[2] = context.recentTopicTokens[1];
        context.recentTopicTokens[1] = context.recentTopicTokens[0];
        context.recentTopicTokens[0] = uint16(inputTokens[0]);

        // Track class history
        if (context.classHistory[maxClass] < 255) {
            context.classHistory[maxClass]++;
        }

        // Update total interactions
        if (context.totalInteractions < type(uint16).max) {
            context.totalInteractions++;
        }

        // Update global statistics
        totalClassifications++;
        classDistribution[maxClass]++;

        emit SentimentClassified(msg.sender, maxClass, confidence, inputString, domain);
        return (maxClass, confidence, domain);
    }

    // Additional functions for functionality

    // Getters
    function getTokenVocab(uint256 tokenId) external view returns (string memory) {
        require(tokenId < vocabSize, "Invalid token ID");
        return tokenVocab[tokenId];
    }

    function getWordToToken(string calldata word) external view returns (uint256) {
        return wordToToken[word];
    }

    function getTokenMetadata(uint256 tokenId) external view 
        returns (
            int8 sentiment,
            uint8 flags,
            uint8 category,
            uint8 weight,
            uint8 domainRelevance,
            uint8 secondaryCategory,
            uint16 cooccurrenceCount,
            uint8 contextInfluence
        ) 
    {
        require(tokenId < vocabSize, "Invalid token ID");
        TokenMetadata memory meta = tokenMetadata[tokenId];
        return (
            meta.sentiment,
            meta.flags,
            meta.category,
            meta.weight,
            meta.domainRelevance,
            meta.secondaryCategory,
            meta.cooccurrenceCount,
            meta.contextInfluence
        );
    }

    function getVocabSize() external view returns (uint256) {
        return vocabSize;
    }

    function getTotalClassifications() external view returns (uint256) {
        return totalClassifications;
    }

    function getCorrectPredictions() external view returns (uint256) {
        return correctPredictions;
    }

    function getClassDistribution(uint8 classId) external view returns (uint256) {
        require(classId < CLASS_COUNT, "Invalid class ID");
        return classDistribution[classId];
    }

    function getPhraseCount() external view returns (uint256) {
        return phraseCount;
    }

    // Setters (admin only)
    function setVocab(
        uint256[] calldata tokenIds,
        string[] calldata words,
        int8[] calldata sentiments,
        uint8[] calldata flags,
        uint8[] calldata categories,
        uint8[] calldata weights,
        uint8[] calldata domainRelevance,
        uint8[] calldata secondaryCategories,
        uint8[] calldata contextInfluence
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(tokenIds.length == words.length, "Array length mismatch");
        require(tokenIds.length <= MAX_VOCAB_SIZE, "Too many tokens");
        require(tokenIds.length == sentiments.length, "Array length mismatch");
        require(tokenIds.length == flags.length, "Array length mismatch");
        require(tokenIds.length == categories.length, "Array length mismatch");
        require(tokenIds.length == weights.length, "Array length mismatch");
        require(tokenIds.length == domainRelevance.length, "Array length mismatch");
        require(tokenIds.length == secondaryCategories.length, "Array length mismatch");
        require(tokenIds.length == contextInfluence.length, "Array length mismatch");

        for (uint256 i = 0; i < tokenIds.length; i++) {
            require(weights[i] > 0 && weights[i] <= 10, "Invalid weight");
            require(contextInfluence[i] > 0 && contextInfluence[i] <= 10, "Invalid context influence");
            require(domainRelevance[i] < MAX_DOMAIN_COUNT, "Invalid domain");

            tokenVocab[tokenIds[i]] = words[i];
            wordToToken[words[i]] = tokenIds[i];

            tokenMetadata[tokenIds[i]] = TokenMetadata({
                sentiment: sentiments[i],
                flags: flags[i],
                category: categories[i],
                weight: weights[i],
                domainRelevance: domainRelevance[i],
                secondaryCategory: secondaryCategories[i],
                cooccurrenceCount: 0,
                contextInfluence: contextInfluence[i]
            });

            if (tokenIds[i] >= vocabSize) vocabSize = tokenIds[i] + 1;
        }

        emit VocabUpdated(tokenIds.length, msg.sender);
    }

    // Pause functions
    function pause() external onlyRole(ADMIN_ROLE) { _pause(); }
    function unpause() external onlyRole(ADMIN_ROLE) { _unpause(); }
}
