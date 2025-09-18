import wikipediaapi
import os

wiki_wiki=wikipediaapi.Wikipedia(language='en',
                                 user_agent='rag-wiki-assistant/1.0 (lteferi3993@gmail.com)')

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

titles = [
    "Machine learning",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Semi-supervised learning",
    "Feature engineering",
    "Cross-validation (statistics)",
    "Bias-variance tradeoff",
    "Overfitting",
    "Regularization (mathematics)",
    "Decision tree learning",
    "Random forest",
    "Support vector machine",
    "k-nearest neighbors algorithm",
    "Naive Bayes classifier",
    "Gradient boosting",
    "XGBoost",
    "Deep learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (machine learning model)",
    "Large language model",
    "Fine-tuning (machine learning)",
    "Transfer learning",
    "Prompt engineering",
    "Embedding (machine learning)",
    "Few-shot learning",
    "Self-supervised learning",
    "Natural language processing",
    "Generative pre-trained transformer"
]

for title in titles:
    page = wiki_wiki.page(title)

    if page.exists():
        try:
            text = page.text
            with open(os.path.join(output_dir, f"{title}.txt"), "w", encoding="utf-8") as f:
                f.write(page.text)
            print(f"Saved {title}.txt file in the {output_dir} directory")
        except Exception as e:
            print(f"Error fetching page: {e}")
    else:
        print('Page not found')


