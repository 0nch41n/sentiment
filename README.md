# OnchainNLP: Onchain Context-Aware Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OnchainNLP is a sophisticated onchain sentiment analyzer implemented entirely in Solidity. It brings advanced context-aware sentiment analysis to blockchain environments by leveraging vector embeddings and user adaptation within the constraints of the EVM.

## 🧠 Architecture Overview

This contract implements a novel approach to onchain sentiment analysis through a context-aware architecture with domain adaptation:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  ┌─────────────┐    ┌─────────────┐             │
│  │  Token      │    │  Context    │             │
│  │  Embeddings │    │  Embeddings │             │
│  └─────────────┘    └─────────────┘             │
│          │                 │                    │
│          ▼                 ▼                    │
│  ┌──────────────────────────────────────────┐   │
│  │                                          │   │
│  │      Semantic Similarity Networks        │   │
│  │                                          │   │
│  └──────────────────────────────────────────┘   │
│          │                 │                    │
│          ▼                 ▼                    │
│  ┌─────────────┐    ┌─────────────┐             │
│  │   User      │    │  Domain     │             │
│  │  Context    │    │  Awareness  │             │
│  └─────────────┘    └─────────────┘             │
│          │                 │                    │
│          ▼                 ▼                    │
│  ┌──────────────────────────────────────────┐   │
│  │                                          │   │
│  │  Multi-Factor Confidence-Scored Output   │   │
│  │                                          │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

## ✨ Key Features

- **Vector Embeddings**: Uses 24-dimensional semantic embeddings and 8-dimensional context embeddings for token representation
- **Context Awareness**: Maintains user history and topic continuity for enhanced contextual understanding
- **Domain-Specific Analysis**: Adapts sentiment analysis across 10 different domains with specialized modifiers
- **Multi-Word Recognition**: Detects phrases of up to 4 words for improved semantic understanding
- **Co-occurrence Tracking**: Builds semantic relationships between tokens based on usage patterns
- **User Adaptation**: Learns from user history and adjusts sentiment classification accordingly
- **Confidence Scoring**: Provides confidence metrics for classification reliability
- **Role-Based Access Control**: Implements sophisticated permission system with distinct roles for administration, training, and feedback

## 🔬 Technical Implementation

### Advanced Sentiment Analysis Components

The contract implements sophisticated NLP operations through several mechanisms:

1. **Multi-Factor Classification**
   ```solidity
   function classifySentiment(uint256[] calldata inputTokens)
       external
       whenNotPaused
       returns (uint8 classId, int256 confidence, uint8 domain)
   ```
   This function performs comprehensive sentiment analysis by aggregating embeddings, calculating class scores with context awareness, applying domain-specific modifiers, and generating confidence metrics.

2. **Token Similarity Calculation**
   ```solidity
   function _calculateTokenSimilarity(
       uint256 tokenA, 
       uint256 tokenB, 
       bool includeContext
   ) internal view returns (int256)
   ```
   Implements a sophisticated similarity mechanism that considers semantic embeddings, context embeddings, category relationships, domain relevance, and co-occurrence patterns.

3. **Domain Detection**
   ```solidity
   function _determinePrimaryDomain(uint256[] memory tokens) internal view returns (uint8)
   ```
   Analyzes token domain relevance to determine the most appropriate domain for contextual adaptation.

### Data Structures

The contract employs several advanced data structures:

1. **Token Metadata**
   ```solidity
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
   ```

2. **User Context**
   ```solidity
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
   ```

3. **Domain Modifier**
   ```solidity
   struct DomainModifier {
       int8[CLASS_COUNT] classBias;
       uint8 intensity;
   }
   ```

4. **Phrase Recognition**
   ```solidity
   struct Phrase {
       string[] words;
       uint256 tokenId;
       uint8 primaryWordIndex;
   }
   ```

## 📊 Performance & Constraints

The model operates within the following constraints:

- Vocabulary size: up to 1,024 tokens
- Embedding dimensions: 24 semantic + 8 context dimensions
- Max input length: 16 tokens
- Sentiment classes: 7 classes (very negative to very positive)
- Semantic categories: 9 primary categories

Despite these constraints, the model achieves sophisticated sentiment analysis through multi-factor scoring, context awareness, and domain adaptation.

## 🚀 Usage Examples

### Sentiment Classification

```solidity
// Tokenize input text (implementation not shown)
uint256[] memory tokens = tokenizer.tokenize("This product exceeded my expectations!");

// Classify sentiment
(uint8 sentimentClass, int256 confidence, uint8 domain) = sentimentAnalyzer.classifySentiment(tokens);

// sentimentClass will be a value from 0-6 (very negative to very positive)
// confidence indicates classification certainty (0-1000)
// domain indicates detected topic domain
```

### Adding Token Embeddings

```solidity
// Add token embeddings for the word "excellent" with positive sentiment
uint256[] memory tokenIds = new uint256[](1);
tokenIds[0] = 42;
string[] memory words = new string[](1);
words[0] = "excellent";
int8[] memory sentiments = new int8[](1);
sentiments[0] = 4; // Very positive sentiment
uint8[] memory flags = new uint8[](1);
flags[0] = SentimentMiniLLM.FLAG_POSITIVE | SentimentMiniLLM.FLAG_INTENSE;
uint8[] memory categories = new uint8[](1);
categories[0] = SentimentMiniLLM.CATEGORY_ATTRIBUTE;
uint8[] memory weights = new uint8[](1);
weights[0] = 8; // High importance weight
uint8[] memory domainRelevance = new uint8[](1);
domainRelevance[0] = SentimentMiniLLM.DOMAIN_GENERAL;
uint8[] memory secondaryCategories = new uint8[](1);
secondaryCategories[0] = SentimentMiniLLM.CATEGORY_EMOTION;
uint8[] memory contextInfluence = new uint8[](1);
contextInfluence[0] = 7; // High context influence

sentimentAnalyzer.setVocab(
    tokenIds,
    words,
    sentiments,
    flags,
    categories,
    weights,
    domainRelevance,
    secondaryCategories,
    contextInfluence
);
```

## 🔧 Technical Requirements

- Solidity ^0.8.19
- OpenZeppelin Contracts (AccessControl, Pausable)
- Custom blockspace or L2 environment recommended due to computational intensity

## 📈 Future Development

- Implement feedback-based embedding adjustments
- Expand vocabulary capacity with efficient storage
- Add support for more complex multi-word expressions
- Implement fine-grained domain adaptation techniques
- Integrate with decentralized reputation systems

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Academic Context

This implementation draws inspiration from NLP concepts in sentiment analysis and context modeling, adapted for the unique constraints of blockchain environments. The vector embedding approach draws from techniques presented in various papers on distributional semantics.

---

*Note: This is an experimental research project demonstrating the intersection of neural language understanding and blockchain technology. While it implements conceptual elements of modern NLP, it operates at a significantly reduced scale compared to traditional off-chain sentiment analyzers.*
