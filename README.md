# 🛠️ Conversational Memory QA System with Scraping

This project automatically scrapes company website data and then builds a **memory-augmented chatbot** using open-source models.  
It uses:
- **Playwright** for headless web scraping.
- **TinyLlama-1.1B-Chat-v1.0** as the conversational LLM.
- **LlamaIndex** for semantic retrieval.
- A custom **LSTM memory network** to track conversation similarity.
- Manual **approval** before storing conversation history.

---

## 🚀 Features

- **Web Scraping**:
  - Scrapes text, services, customers, blog posts, and tone of voice.
  - Saves results into structured JSON files in the `data/` folder.
- **Conversational Chatbot**:
  - Loads scraped content into a vector index.
  - Answers user queries based on the knowledge base.
  - Tracks conversation memory and warns if a new answer is too repetitive.
  - Only stores approved exchanges into conversation history.
  - Utilizes EOS token stopping to reduce generation randomness.

---

## 📂 Project Structure

```
.
├── scraping.py          # Scrapes data from the target website
├── chatbot.py           # Memory-augmented chatbot over scraped data
├── data/                # Scraped content (auto-generated)
├── storage/             # Persistent vector index (auto-generated)
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
```

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/conversational-memory-qa.git
   cd conversational-memory-qa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   playwright install
   ```

---

## ⚙️ Usage

### 1. Scrape Website Content

Run the scraper to collect company data:
```bash
python scraping.py
```
This will populate the `data/` directory with several JSON files.

---

### 2. Start the Chatbot

After scraping, launch the chatbot:
```bash
python chatbot.py
```

You can now interact with the knowledge base conversationally!

Example:

```
Ask something (or type 'exit'): What services does the company offer?

Generated Answer:
[Model-generated description of services]

Approve this answer? (y/n): y
👌 Response added to conversation memory.
```

---

## 📊 Conversation Memory

- **LSTM-based embeddings** are used to encode prior conversation turns.
- **Cosine similarity** is computed between the current response and historical memory.
- If the similarity is **above 0.7**, a **warning** is shown.
- Only **approved answers** are added to memory, allowing controlled memory growth.

---

## 🧠 Models Used

- **TinyLlama-1.1B-Chat-v1.0** — lightweight and fast conversational LLM.
- **all-MiniLM-L6-v2** — sentence-transformer for embeddings.
- **Custom LSTM Network** — handles long-term memory and conversation tracking.

---

# ✨ Quick Start Commands

```bash
# Install
pip install -r requirements.txt
playwright install

# Scrape data
python scraping.py

# Start chatbot
python chatbot.py
```


