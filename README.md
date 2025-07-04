# Chatgpt-query
build a  pipeline that analyzes press releases and predicts their likely impact on stock price movement

# 📊 Press Release Impact Rating using Duck.ai (GPT-4o mini)

This project analyzes financial press releases and predicts their likely impact on stock prices using a two-step large language model (LLM) pipeline via [Duck.ai](https://duck.ai).

- ✅ Categorizes press releases using **LLaMA 3** and keyword-based logic
- ✅ Uses **GPT-4o mini** to rate the **market sentiment** (from -1 to 1)
- ✅ Tailors prompts by press release category for more reliable results

---

## 🔍 How It Works

1. **Categorization Stage**
   - Press releases are classified into one of six categories:
     - `resultats_financiers` (financial results)
     - `dividendes` (dividends)
     - `projets_strategiques` (strategic projects)
     - `changement_direction` (executive changes)
     - `partenariat` (partnerships)
     - `litiges_reglementations` (lawsuits & regulations)
   - Classification is powered by **LLaMA 3 7B** running locally using keyword-matching heuristics.

2. **Sentiment Rating Stage**
   - Each categorized press release is passed to **GPT-4o mini**.
   - A category-specific prompt guides the LLM to rate the semantic influence of the press release on the stock price.
   - The rating is a float in the range `[-1, 1]`:
     - `-1` → Strong **downward** pressure on stock price
     - `0` → **Neutral** or uncertain impact
     - `1` → Strong **upward** pressure on stock price

---
