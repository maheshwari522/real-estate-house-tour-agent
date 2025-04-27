```markdown
# 🏡 House Tour Voice AI Assistant

This project is a Voice-Controlled AI Assistant that:
- Loads your house information from text files 📄
- Allows you to **ask questions by speaking** 🎙️
- Uses **Retrieval-Augmented Generation (RAG)** to generate natural answers 🧠
- **Speaks back the answers** using ElevenLabs voice output 🔊

---

## 📦 Setup Instructions

### 1. Create and Activate Python Environment

```bash
python3 -m venv housetour-env
source housetour-env/bin/activate   # Mac/Linux
# or
housetour-env\Scripts\activate      # Windows
```

---

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

(You must have a `requirements.txt` file ready — I can help you generate it if needed!)

---

### 3. Create a `.env` File

Create a file named `.env` in your project root directory, and add the following:

```bash
OPENAI_API_KEY=your-openai-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
GROQ_API_KEY=your-groq-api-key-here
```

✅ This keeps your API keys secure and separate from code.

---

## 🚀 Execution Flow

### Step 1: Build the Knowledgebase

Run:

```bash
python knowledgebase.py
```

- **Important:**  
  Before running, edit `knowledgebase.py` to set:

```python
main_folder = "<Your-path>/HouseTour/Images"
```

---

### Step 2: Create House Tour Script and Audio

Run:

```bash
python housetour.py
```

- **Important:**  
  Before running, set the following paths inside `housetour.py`:

```python
property_details_path = "<Your-path>/HouseTour/Images/property_details.txt"
room_descriptions_path = "<Your-path>/HouseTour/Images/room_description.txt"
output_script_path = "<Your-path>/HouseTour/Images/house_tour_script.txt"
output_audio_path = "<Your-path>/HouseTour/Images/house_tour_audio.mp3"
```

---

### Step 3: Run House Tour RAG (Text-based Retrieval QA)

Run:

```bash
python housetourrag.py
```

- **Important:**  
  Before running, make sure you set the folder paths in `housetourrag.py`:

```python
folders = ["<Your-path>/HouseTour/Images"]
```

---

### Step 4: Run House Tour Voice Assistant (Voice-based RAG)

Before running, **set API keys in terminal** (optional if already using .env):

```bash
export ELEVENLABS_API_KEY=your-elevenlabs-api-key
export OPENAI_API_KEY=your-openai-api-key
export GROQ_API_KEY=your-groq-api-key
```

Then run:

```bash
python housetourragwithvoiceagent.py
```

✅ This launches a full voice bot:
- Speak your question (like "Tell me about the kitchen")
- It listens, understands, answers, and speaks back!

---

## 🛠️ Project Folder Structure

```bash
HouseTour/
├── Images/
│   ├── property_details.txt
│   ├── room_description.txt
│   └── (other .txt files)
├── knowledgebase.py
├── housetour.py
├── housetourrag.py
├── housetourragwithvoiceagent.py
├── config.py
├── .env
└── requirements.txt
```

---

## ⚙️ Requirements

- Python 3.9+
- OpenAI API Key
- ElevenLabs API Key
- Groq API Key
- Internet Connection

---

## 🌟 Technologies Used

| Purpose | Technology |
|:--|:--|
| Vector Search | FAISS |
| Language Model | OpenAI GPT-4 |
| Voice-to-Text (STT) | OpenAI Whisper via Groq |
| Text-to-Voice (TTS) | ElevenLabs |
| Knowledgebase | LangChain RetrievalQA |

---
```
